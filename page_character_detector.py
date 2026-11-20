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
    les gouttières de séparation claires (papier/fond).
    Gère aussi bien les planches multi-cases que les splash pages pleine page.
    Retourne la liste des boîtes [x1, y1, x2, y2] triées dans l'ordre de lecture.
    """
    orig_w, orig_h = img.size
    analysis_w = 600
    analysis_h = int(analysis_w * orig_h / orig_w)
    thumb = img.convert("L").resize((analysis_w, analysis_h), Image.Resampling.BILINEAR)
    arr = np.array(thumb)

    # Le fond de papier des gouttières est dans les percentiles élevés (>= 80%)
    bg_thresh = max(190, int(np.percentile(arr, 85)))
    is_bg = arr >= bg_thresh

    # 1. Détection des bandes horizontales (strips)
    row_bg_ratio = is_bg[:, int(analysis_w * 0.05):int(analysis_w * 0.95)].mean(axis=1)
    strips = []
    in_strip = False
    start_y = 0

    for y in range(analysis_h):
        if row_bg_ratio[y] < 0.78:
            if not in_strip:
                in_strip = True
                start_y = y
        else:
            if in_strip:
                in_strip = False
                if (y - start_y) > 0.08 * analysis_h:
                    strips.append((start_y, y))

    if in_strip and (analysis_h - start_y) > 0.08 * analysis_h:
        strips.append((start_y, analysis_h))

    # Si aucune séparation horizontale claire n'est trouvée (ex: splash page pleine page)
    if not strips or len(strips) <= 1:
        return [(0, 0, orig_w, orig_h)]

    scale_x = orig_w / analysis_w
    scale_y = orig_h / analysis_h
    panels = []

    # 2. Découpage horizontal en cases à l'intérieur de chaque bande
    for (y1, y2) in strips:
        strip_bg = is_bg[y1:y2, :]
        col_bg_ratio = strip_bg.mean(axis=0)

        in_panel = False
        start_x = 0
        for x in range(analysis_w):
            if col_bg_ratio[x] < 0.78:
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
        candidates: List[Tuple[Image.Image, List[int]]] = []

        # A) Détection via YOLO si disponible (seuil abaissé pour cartoon)
        if self.yolo_model:
            try:
                results = self.yolo_model(panel_img, conf=0.10, verbose=False)
                for r in results:
                    for b in r.boxes:
                        name = self.yolo_model.names[int(b.cls[0])]
                        conf = float(b.conf[0])
                        if name in LIVING_CLASSES:
                            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                            if (x2 - x1) >= 30 and (y2 - y1) >= 30:
                                bw, bh = x2 - x1, y2 - y1
                                x1_m = max(0, int(x1 - 0.15 * bw))
                                y1_m = max(0, int(y1 - 0.15 * bh))
                                x2_m = min(pw, int(x2 + 0.15 * bw))
                                y2_m = min(ph, int(y2 + 0.15 * bh))
                                candidates.append((
                                    panel_img.crop((x1_m, y1_m, x2_m, y2_m)),
                                    [x1_m, y1_m, x2_m, y2_m]
                                ))
            except Exception:
                pass

        # B) Propositions multi-échelles pour les cases larges ou splash (quand YOLO a un faible rappel cartoon)
        if len(candidates) < 3 and pw >= 300 and ph >= 300:
            # Sous-découpages typiques où se trouvent les personnages (tiers inférieurs et quadrants)
            sub_proposals = [
                # Moitié gauche et droite du bas
                (int(pw * 0.08), int(ph * 0.55), int(pw * 0.50), int(ph * 0.96)),
                (int(pw * 0.45), int(ph * 0.55), int(pw * 0.92), int(ph * 0.96)),
                # Tiers gauche, centre, droite
                (int(pw * 0.10), int(ph * 0.62), int(pw * 0.38), int(ph * 0.95)),
                (int(pw * 0.32), int(ph * 0.62), int(pw * 0.60), int(ph * 0.95)),
                (int(pw * 0.58), int(ph * 0.62), int(pw * 0.90), int(ph * 0.95)),
            ]
            for bx1, by1, bx2, by2 in sub_proposals:
                if bx2 > bx1 and by2 > by1:
                    candidates.append((panel_img.crop((bx1, by1, bx2, by2)), [bx1, by1, bx2, by2]))

        # C) Si aucun candidat, analyser la case entière
        if not candidates and pw >= 60 and ph >= 60:
            candidates.append((panel_img, [0, 0, pw, ph]))

        # Identification via DINOv2
        detections = []
        seen_heroes_in_panel = set()

        for crop, box in candidates:
            preds = self.matcher.identify(crop, top_k=2, aggregate_by_hero=True)
            if preds and preds[0]["score"] >= min_score:
                h = preds[0]["hero"]
                # Déduplication : on garde la meilleure boîte pour chaque héros dans cette case
                if h not in seen_heroes_in_panel:
                    seen_heroes_in_panel.add(h)
                    detections.append({
                        "hero": h,
                        "score": preds[0]["score"],
                        "reference": preds[0]["best_reference"],
                        "box": box
                    })

        return detections

    def analyze_page(self, img_input, min_score: float = 0.70) -> Dict[str, any]:
        """
        Analyse une page de BD complète :
        1. Découpe en cases
        2. Détection des personnages par case (avec coordonnées absolues)
        3. Consolidation et classement des apparitions
        """
        if isinstance(img_input, str):
            with Image.open(img_input) as f:
                img = f.convert("RGB")
        else:
            img = img_input.convert("RGB")

        panels = segment_panels_adaptive(img)
        panel_detections = []
        all_detections = []
        hero_counts = {}
        hero_max_scores = {}

        for p_idx, (x1, y1, x2, y2) in enumerate(panels, start=1):
            panel_crop = img.crop((x1, y1, x2, y2))
            chars_in_panel = self.detect_in_panel(panel_crop, min_score=min_score)

            seen_in_panel = set()
            for det in chars_in_panel:
                h = det["hero"]
                score = det["score"]
                bx1, by1, bx2, by2 = det["box"]
                # Coordonnées absolues sur la planche complète
                abs_box = [x1 + bx1, y1 + by1, x1 + bx2, y1 + by2]

                all_detections.append({
                    "hero": h,
                    "score": score,
                    "panel_index": p_idx,
                    "panel_box": [x1, y1, x2, y2],
                    "box": abs_box,
                    "reference": det["reference"]
                })

                if h not in seen_in_panel:
                    seen_in_panel.add(h)
                    hero_counts[h] = hero_counts.get(h, 0) + 1
                    hero_max_scores[h] = max(hero_max_scores.get(h, 0.0), score)

            panel_detections.append({
                "panel_index": p_idx,
                "box": [x1, y1, x2, y2],
                "characters": chars_in_panel
            })

        sorted_heroes = sorted(
            hero_counts.keys(),
            key=lambda h: (hero_counts[h], hero_max_scores[h]),
            reverse=True
        )

        return {
            "panels": panels,
            "panel_detections": panel_detections,
            "all_detections": all_detections,
            "hero_counts": hero_counts,
            "hero_max_scores": hero_max_scores,
            "sorted_heroes": sorted_heroes,
            "top_hero": sorted_heroes[0] if sorted_heroes else "--"
        }
