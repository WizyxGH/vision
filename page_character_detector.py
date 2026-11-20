"""
Détecteur et identificateur de personnages Disney à l'échelle d'une planche de BD complète.
Découpe les cases, localise les personnages, extrait les embeddings DINOv2 et consolide
la liste des personnages présents (xapp) et le héro probable de la page.
"""

import os
import json
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from search_character import CharacterMatcher, DEFAULT_INDEX_PATH

# Classes COCO qui correspondent généralement aux personnages vivants dans un cartoon
LIVING_CLASSES = {"person", "bird", "cat", "dog", "teddy bear", "horse", "sheep", "cow", "bear"}


def segment_panels_adaptive(img: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Découpe une planche de BD en cases individuelles (panels) en détectant
    les gouttières de séparation basées sur la couleur réelle des marges de la page.
    Retourne la liste des boîtes [x1, y1, x2, y2] triées dans l'ordre de lecture.
    """
    orig_w, orig_h = img.size
    # Redimensionnement temporaire pour une analyse vectorielle ultra-rapide
    analysis_w = 600
    analysis_h = int(analysis_w * orig_h / orig_w)
    thumb = img.resize((analysis_w, analysis_h), Image.Resampling.BILINEAR)
    arr = np.array(thumb, dtype=np.int16)

    # 1. Échantillonner la couleur de fond sur le pourtour extérieur
    m = max(3, analysis_w // 40)
    border = np.concatenate([
        arr[:, :m].reshape(-1, 3),
        arr[:, -m:].reshape(-1, 3),
        arr[:m, :].reshape(-1, 3),
        arr[-m:, :].reshape(-1, 3)
    ])
    bg_color = np.median(border, axis=0)

    # 2. Masque binaire des pixels proches du fond (gouttières et marges)
    diff = np.abs(arr - bg_color).sum(axis=2)
    is_bg = diff < 48

    # 3. Détection des bandes horizontales (tiers)
    row_bg_ratio = is_bg[:, int(analysis_w * 0.05):int(analysis_w * 0.95)].mean(axis=1)
    strips = []
    in_strip = False
    start_y = 0

    for y in range(analysis_h):
        if row_bg_ratio[y] < 0.88:
            if not in_strip:
                in_strip = True
                start_y = y
        else:
            if in_strip:
                in_strip = False
                if (y - start_y) > 0.07 * analysis_h:
                    strips.append((start_y, y))

    if in_strip and (analysis_h - start_y) > 0.07 * analysis_h:
        strips.append((start_y, analysis_h))

    # Si aucune bande distincte n'est trouvée, considérer la page comme une case unique
    if not strips:
        return [(0, 0, orig_w, orig_h)]

    scale_x = orig_w / analysis_w
    scale_y = orig_h / analysis_h
    panels = []

    # 4. Découpage horizontal en cases à l'intérieur de chaque bande
    for (y1, y2) in strips:
        strip_bg = is_bg[y1:y2, :]
        col_bg_ratio = strip_bg.mean(axis=0)

        in_panel = False
        start_x = 0
        for x in range(analysis_w):
            if col_bg_ratio[x] < 0.85:
                if not in_panel:
                    in_panel = True
                    start_x = x
            else:
                if in_panel:
                    in_panel = False
                    if (x - start_x) > 0.08 * analysis_w:
                        panels.append((
                            int(start_x * scale_x),
                            int(y1 * scale_y),
                            int(x * scale_x),
                            int(y2 * scale_y)
                        ))
        if in_panel and (analysis_w - start_x) > 0.08 * analysis_w:
            panels.append((
                int(start_x * scale_x),
                int(y1 * scale_y),
                int(analysis_w * scale_x),
                int(y2 * scale_y)
            ))

    return panels if panels else [(0, 0, orig_w, orig_h)]


class PageCharacterDetector:
    def __init__(self, index_path: str = DEFAULT_INDEX_PATH, yolo_model_path: str = "yolov8n.pt"):
        self.matcher = CharacterMatcher(index_path=index_path)
        self.yolo_model = YOLO(yolo_model_path) if HAS_YOLO else None

    def detect_in_panel(self, panel_img: Image.Image, min_score: float = 0.70) -> List[Dict]:
        """Détecte et identifie les personnages présents à l'intérieur d'une case."""
        pw, ph = panel_img.size
        candidates = []

        # A) Détection via YOLO si disponible
        if self.yolo_model:
            results = self.yolo_model(panel_img, conf=0.20, verbose=False)
            for r in results:
                for b in r.boxes:
                    name = self.yolo_model.names[int(b.cls[0])]
                    conf = float(b.conf[0])
                    if name in LIVING_CLASSES:
                        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                        # Filtrer les boîtes trop petites
                        if (x2 - x1) >= 35 and (y2 - y1) >= 35:
                            # Marge de 15% pour inclure la coiffe / le vêtement
                            bw, bh = x2 - x1, y2 - y1
                            x1_m = max(0, int(x1 - 0.15 * bw))
                            y1_m = max(0, int(y1 - 0.15 * bh))
                            x2_m = min(pw, int(x2 + 0.15 * bw))
                            y2_m = min(ph, int(y2 + 0.15 * bh))
                            candidates.append(panel_img.crop((x1_m, y1_m, x2_m, y2_m)))

        # B) Si aucun personnage n'a été détecté par YOLO (ex: gros plan stylisé), analyser la case entière
        if not candidates and pw >= 60 and ph >= 60:
            candidates.append(panel_img)

        # Identification via DINOv2
        detections = []
        for crop in candidates:
            preds = self.matcher.identify(crop, top_k=2, aggregate_by_hero=True)
            if preds and preds[0]["score"] >= min_score:
                detections.append({
                    "hero": preds[0]["hero"],
                    "score": preds[0]["score"],
                    "reference": preds[0]["best_reference"]
                })

        return detections

    def analyze_page(self, img_input, min_score: float = 0.70) -> Dict[str, any]:
        """
        Analyse une page de BD complète :
        1. Découpe en cases
        2. Détection des personnages par case
        3. Consolidation et classement des apparitions
        """
        if isinstance(img_input, str):
            with Image.open(img_input) as f:
                img = f.convert("RGB")
        else:
            img = img_input.convert("RGB")

        panels = segment_panels_adaptive(img)
        panel_detections = []
        hero_counts = {}
        hero_max_scores = {}

        for p_idx, (x1, y1, x2, y2) in enumerate(panels, start=1):
            panel_crop = img.crop((x1, y1, x2, y2))
            chars_in_panel = self.detect_in_panel(panel_crop, min_score=min_score)

            # Éviter de compter deux fois le même héros dans la même case
            seen_in_panel = set()
            for det in chars_in_panel:
                h = det["hero"]
                score = det["score"]
                if h not in seen_in_panel:
                    seen_in_panel.add(h)
                    hero_counts[h] = hero_counts.get(h, 0) + 1
                    hero_max_scores[h] = max(hero_max_scores.get(h, 0.0), score)

            panel_detections.append({
                "panel_index": p_idx,
                "box": [x1, y1, x2, y2],
                "characters": chars_in_panel
            })

        # Trier les personnages par nombre d'apparitions puis score
        sorted_heroes = sorted(
            hero_counts.keys(),
            key=lambda h: (hero_counts[h], hero_max_scores[h]),
            reverse=True
        )

        main_hero = sorted_heroes[0] if sorted_heroes else ""
        xapp = sorted(sorted_heroes)

        return {
            "total_panels": len(panels),
            "panels": panel_detections,
            "detected_characters": sorted_heroes,
            "hero_counts": hero_counts,
            "hero_max_scores": hero_max_scores,
            "main_hero": main_hero,
            "xapp": xapp
        }


def main():
    parser = argparse.ArgumentParser(description="Analyse une page de BD pour détecter les personnages Disney")
    parser.add_argument("image", help="Chemin vers la page scannée")
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help="Chemin vers l'index DINOv2")
    parser.add_argument("--min-score", type=float, default=0.72, help="Score de similarité cosinus minimum (0.0 à 1.0)")
    parser.add_argument("--json", action="store_true", help="Sortie au format JSON brut")

    args = parser.parse_args()

    detector = PageCharacterDetector(index_path=args.index)
    res = detector.analyze_page(args.image, min_score=args.min_score)

    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    print(f"\n=======================================================")
    print(f" Planche : {os.path.basename(args.image)}")
    print(f" Cases détectées : {res['total_panels']}")
    print(f"=======================================================")
    print(f" Héros principal présumé : {res['main_hero'] or 'Non déterminé'}")
    print(f" Liste d'apparitions (xapp) : {', '.join(res['xapp']) if res['xapp'] else 'Aucun'}\n")

    print("Détail des personnages détectés :")
    for h in res["detected_characters"]:
        count = res["hero_counts"][h]
        max_s = res["hero_max_scores"][h] * 100
        print(f"  • {h:<18} : vu dans {count:2d} case(s) (score max: {max_s:5.1f}%)")


if __name__ == "__main__":
    main()
