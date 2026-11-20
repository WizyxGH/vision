"""
apps/vision/dialogue_speaker_extractor.py

Moteur d'extraction et d'attribution des dialogues par case et par personnage.
Combine :
1. La segmentation adaptative des cases (Panels).
2. La détection et reconnaissance des personnages (YOLO + DINOv2 Prototypes Inducks).
3. L'OCR des lignes de texte (PaddleOCR) regroupées en bulles / blocs cohérents.
4. L'attribution du locuteur (Speaker Attribution) à l'intérieur de la case.
5. Le typage sans suppression (dialogue, narration, onomatopée, titre, décor).
"""

import os
import re
import math
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from PIL import Image
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Importer torch en premier pour éviter le conflit OpenMP/DLL sous Windows
import torch

try:
    import paddle
    paddle.device.set_device('cpu')
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except Exception:
    HAS_PADDLE = False

from page_character_detector import segment_panels_adaptive, PageCharacterDetector, LIVING_CLASSES
from multimodal_character_fusion import DIALOGUE_PATTERNS


# Mot-clés narratifs courants (italien, français, anglais)
NARRATIVE_MARKERS = [
    r"^(?:mentre|nel frattempo|intanto|lungo il|poco dopo|il giorno dopo|all'improvviso|intanto che)\b",
    r"^(?:pendant ce temps|le lendemain|soudain|quelques instants plus tard|tandis que)\b",
    r"^(?:meanwhile|later|suddenly|the next day|meanwhile,)\b",
]
NARRATIVE_RE = re.compile("|".join(NARRATIVE_MARKERS), re.IGNORECASE)

# Mot-clés de métadonnées, titres, codes Inducks et signatures
HEADER_METADATA_PATTERNS = [
    r"^\s*walt\s*disney\s*$",
    r"^\s*(?:ip|i|d|fr|de|nl|gr|us|s|w)\s*[-/]?\s*\d+[-/]\w+\s*$",
    r"^\s*topolino\s*\d*\s*$",
    r"^\s*\d+\s*$",
    r"^\s*(?:continua|fine|segue|à suivre|fin)\s*$",
]
HEADER_METADATA_RE = re.compile("|".join(HEADER_METADATA_PATTERNS), re.IGNORECASE)
SOUND_EFFECT_RE = re.compile(
    r"\b(?:boom|bam|clac|crash|slam|gulp|gloups|bang|pant|sigh|sob|argh|gasp|swoosh|wrooom|vrooom|smack|ring|dring|drin|toc|knock|snif|splash|plop|roar|yikes|oof|ouch)\b",
    re.IGNORECASE
)


def cluster_text_lines_into_bubbles(lines: List[Dict], max_y_dist_ratio: float = 1.3) -> List[Dict]:
    """
    Regroupe les lignes d'OCR proches spatialement en blocs cohérents (bulles de dialogue).
    Conserve toutes les lignes et ordonne le texte du haut vers le bas, de gauche à droite.
    """
    if not lines:
        return []

    # Trier d'abord par position verticale Y
    sorted_lines = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    clusters = []

    for line in sorted_lines:
        lx1, ly1, lx2, ly2 = line["bbox"]
        lh = max(12, ly2 - ly1)
        assigned = False

        for c in clusters:
            cx1, cy1, cx2, cy2 = c["bbox"]
            clh = c["avg_height"]

            # Vérifier la proximité verticale et le chevauchement horizontal
            vert_dist = ly1 - cy2
            max_allowed_dist = max(lh, clh) * max_y_dist_ratio

            # Chevauchement horizontal ou proximité horizontale
            horiz_overlap = max(0, min(lx2, cx2) - max(lx1, cx1))
            horiz_dist = max(0, max(lx1, cx1) - min(lx2, cx2))
            min_width = min(lx2 - lx1, cx2 - cx1)

            # Même bulle si les lignes se suivent verticalement et s'alignent horizontalement
            if -clh * 0.5 <= vert_dist <= max_allowed_dist and (horiz_overlap > 0.2 * min_width or horiz_dist < 40):
                c["lines"].append(line)
                c["bbox"] = (min(cx1, lx1), min(cy1, ly1), max(cx2, lx2), max(cy2, ly2))
                c["avg_height"] = sum(l["bbox"][3] - l["bbox"][1] for l in c["lines"]) / len(c["lines"])
                assigned = True
                break

        if not assigned:
            clusters.append({
                "lines": [line],
                "bbox": (lx1, ly1, lx2, ly2),
                "avg_height": lh
            })

    # Construire les blocs de bulles finaux
    blocks = []
    for c in clusters:
        # Trier les lignes à l'intérieur de la bulle par Y croissant
        c_lines = sorted(c["lines"], key=lambda l: l["bbox"][1])
        merged_text = " ".join(l["text"].strip() for l in c_lines if l["text"].strip())
        avg_conf = sum(l["conf"] for l in c_lines) / max(1, len(c_lines))
        bx1, by1, bx2, by2 = c["bbox"]
        blocks.append({
            "text": merged_text,
            "confidence": round(avg_conf, 3),
            "bbox": (bx1, by1, bx2, by2),
            "center": ((bx1 + bx2) / 2, (by1 + by2) / 2),
            "bottom_center": ((bx1 + bx2) / 2, by2),
            "line_count": len(c_lines)
        })

    # Trier les blocs de bulles dans l'ordre de lecture (haut-gauche vers bas-droite)
    blocks.sort(key=lambda b: (b["bbox"][1] // 60, b["bbox"][0]))
    return blocks


def classify_text_block(block: Dict, panel_bbox: Tuple[int, int, int, int], is_first_panel: bool = False) -> str:
    """
    Détermine la nature du bloc textuel sans jamais le supprimer :
    - 'header' : titre de journal / page (ex: TOPOLINO 11)
    - 'narrative' : récitatif / cartouche narratif
    - 'sound_effect' : onomatopée
    - 'speech' : bulle de dialogue
    """
    text = block["text"].strip()
    bx1, by1, bx2, by2 = block["bbox"]
    px1, py1, px2, py2 = panel_bbox
    bw, bh = bx2 - bx1, by2 - by1
    pw, ph = px2 - px1, py2 - py1

    # 1. En-tête de page, titre, signature ou code histoire
    clean_no_space = re.sub(r"\s+", "", text.lower())
    if "waltdisney" in clean_no_space or HEADER_METADATA_RE.search(text):
        return "header"

    if is_first_panel and by1 < py1 + 80 and ("topolino" in text.lower() or text.isdigit()):
        return "header"

    # 2. Cartouche narratif (reconnaissance de formulation ou cartouche large en haut/bas de case)
    if NARRATIVE_RE.search(text):
        return "narrative"

    if (by1 <= py1 + 40 or by2 >= py2 - 40) and bw >= pw * 0.65 and bh <= ph * 0.25:
        return "narrative"

    # 3. Onomatopées (courtes, souvent majuscules isolées, point d'exclamation fort, ou mots d'effet)
    upper_letters = re.findall(r"[A-ZÀ-Ý]", text)
    if len(text) <= 12 and (len(upper_letters) >= len(text) * 0.7 or SOUND_EFFECT_RE.search(text)):
        if text.endswith("!") or SOUND_EFFECT_RE.search(text):
            return "sound_effect"

    # 4. Par défaut : bulle de dialogue
    return "speech"


class ComicDialogueSpeakerExtractor:
    """
    Moteur de fusion qui analyse une planche de BD complète et extrait
    l'intégralité du texte structuré par case, avec attribution du locuteur.
    """

    def __init__(self, lang: str = "it"):
        self.lang = lang
        self.char_detector = PageCharacterDetector()
        if HAS_PADDLE:
            self.ocr = PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=False, show_log=False)
        else:
            self.ocr = None

    def process_page(self, img_input: Any) -> Dict[str, Any]:
        if isinstance(img_input, (str, Path)):
            img = Image.open(str(img_input)).convert("RGB")
            page_name = Path(img_input).name
        else:
            img = img_input.convert("RGB")
            page_name = "image"

        orig_w, orig_h = img.size

        # Étape 1 : Découpage adaptatif des cases
        panels = segment_panels_adaptive(img)

        # Étape 2 : OCR complet de la planche
        raw_ocr_lines = []
        if self.ocr:
            ocr_res = self.ocr.ocr(np.array(img), cls=False)
            if ocr_res and ocr_res[0]:
                for line in ocr_res[0]:
                    box_pts, (text, conf) = line
                    bx1 = min(pt[0] for pt in box_pts)
                    by1 = min(pt[1] for pt in box_pts)
                    bx2 = max(pt[0] for pt in box_pts)
                    by2 = max(pt[1] for pt in box_pts)
                    raw_ocr_lines.append({
                        "text": text,
                        "conf": conf,
                        "bbox": (int(bx1), int(by1), int(bx2), int(by2))
                    })

        # Étape 3 : Traitement case par case
        structured_panels = []

        for p_idx, p_box in enumerate(panels, 1):
            px1, py1, px2, py2 = p_box
            panel_crop = img.crop(p_box)
            pw, ph = panel_crop.size

            # A) Détection et localisation des personnages dans la case
            char_detections = []
            if self.char_detector.yolo_model:
                yolo_res = self.char_detector.yolo_model(panel_crop, conf=0.18, verbose=False)
                for r in yolo_res:
                    for b in r.boxes:
                        cls_name = self.char_detector.yolo_model.names[int(b.cls[0])]
                        if cls_name in LIVING_CLASSES:
                            cx1, cy1, cx2, cy2 = [int(v) for v in b.xyxy[0].tolist()]
                            if (cx2 - cx1) >= 30 and (cy2 - cy1) >= 30:
                                # Crop avec marge pour DINOv2
                                mx1 = max(0, int(cx1 - 0.15 * (cx2 - cx1)))
                                my1 = max(0, int(cy1 - 0.15 * (cy2 - cy1)))
                                mx2 = min(pw, int(cx2 + 0.15 * (cx2 - cx1)))
                                my2 = min(ph, int(cy2 + 0.15 * (cy2 - cy1)))
                                char_crop = panel_crop.crop((mx1, my1, mx2, my2))
                                preds = self.char_detector.matcher.identify(char_crop, top_k=1, aggregate_by_hero=True)
                                if preds and preds[0]["score"] >= 0.65:
                                    char_detections.append({
                                        "hero": preds[0]["hero"],
                                        "score": round(preds[0]["score"], 3),
                                        "bbox_in_panel": (cx1, cy1, cx2, cy2),
                                        "bbox_in_page": (px1 + cx1, py1 + cy1, px1 + cx2, py1 + cy2),
                                        "head_center": (px1 + (cx1 + cx2) / 2, py1 + cy1 + (cy2 - cy1) * 0.25)
                                    })

            # Si YOLO n'a rien détecté de précis mais qu'un perso est visible sur le panel
            if not char_detections and pw >= 60 and ph >= 60:
                preds = self.char_detector.matcher.identify(panel_crop, top_k=1, aggregate_by_hero=True)
                if preds and preds[0]["score"] >= 0.70:
                    char_detections.append({
                        "hero": preds[0]["hero"],
                        "score": round(preds[0]["score"], 3),
                        "bbox_in_panel": (0, 0, pw, ph),
                        "bbox_in_page": p_box,
                        "head_center": (px1 + pw / 2, py1 + ph * 0.3)
                    })

            # B) Filtrer les lignes OCR qui tombent dans cette case
            p_lines = []
            for l in raw_ocr_lines:
                bx1, by1, bx2, by2 = l["bbox"]
                center_x = (bx1 + bx2) / 2
                center_y = (by1 + by2) / 2
                if px1 <= center_x <= px2 and py1 <= center_y <= py2:
                    p_lines.append(l)

            # C) Regrouper les lignes en bulles
            bubble_blocks = cluster_text_lines_into_bubbles(p_lines)

            # D) Classifier chaque bulle et attribuer le locuteur
            elements = []
            for b in bubble_blocks:
                b_type = classify_text_block(b, p_box, is_first_panel=(p_idx == 1))
                speaker = None
                speaker_confidence = None

                # Lier le locuteur uniquement pour les bulles de dialogue (speech / thought)
                if b_type == "speech" and char_detections:
                    if len(char_detections) == 1:
                        # Un seul personnage présent dans la case -> attribution certaine
                        speaker = char_detections[0]["hero"]
                        speaker_confidence = char_detections[0]["score"]
                    else:
                        # Plusieurs personnages : attribution géométrique (proximité tête/bouche du perso)
                        bx, by = b["bottom_center"]
                        best_char = None
                        min_dist = float("inf")

                        for c in char_detections:
                            hx, hy = c["head_center"]
                            # Distance pondérée : les bulles sont généralement AU-DESSUS de la tête
                            dx = bx - hx
                            dy = hy - by  # positif si la bulle est au-dessus de la tête
                            dist = math.sqrt(dx * dx + (dy * 0.7 if dy > 0 else dy * 2.0) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                best_char = c

                        if best_char:
                            speaker = best_char["hero"]
                            speaker_confidence = best_char["score"]

                elements.append({
                    "type": b_type,
                    "text": b["text"],
                    "speaker": speaker,
                    "speaker_confidence": speaker_confidence,
                    "confidence": b["confidence"],
                    "bbox": b["bbox"]
                })

            structured_panels.append({
                "panel_index": p_idx,
                "bbox": p_box,
                "characters_present": list({c["hero"] for c in char_detections}),
                "elements": elements
            })

        return {
            "page": page_name,
            "page_dimensions": {"width": orig_w, "height": orig_h},
            "panel_count": len(panels),
            "panels": structured_panels
        }


def format_as_screenplay(result: Dict[str, Any]) -> str:
    """Formatte le résultat en script de lecture clair et lisible."""
    lines = []
    lines.append(f"═══════════════════════════════════════════════════════")
    lines.append(f"📖 TRANSCRIPTION STRUCTURÉE : {result['page']}")
    lines.append(f"Dimensions: {result['page_dimensions']['width']}x{result['page_dimensions']['height']} | {result['panel_count']} cases détectées")
    lines.append(f"═══════════════════════════════════════════════════════\n")

    for p in result["panels"]:
        chars = ", ".join(p["characters_present"]) if p["characters_present"] else "Aucun personnage détecté"
        lines.append(f"┌─── CASE #{p['panel_index']} (Personnages : {chars})")

        if not p["elements"]:
            lines.append("│   (Sans texte)")
        else:
            for el in p["elements"]:
                t = el["type"]
                txt = el["text"]
                spk = el["speaker"]

                if t == "speech":
                    spk_label = f"[{spk}]" if spk else "[Personnage inconnu]"
                    lines.append(f"│   💬 {spk_label:14} : « {txt} »")
                elif t == "narrative":
                    lines.append(f"│   📜 [NARRATEUR]   : {txt}")
                elif t == "sound_effect":
                    lines.append(f"│   💥 [EFFET SONORE]: *{txt}*")
                elif t == "header":
                    lines.append(f"│   🏷️  [TITRE/BANDEAU]: {txt}")
                else:
                    lines.append(f"│   📝 [DÉCOR/TEXTE] : {txt}")

        lines.append(f"└───\n")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\starl\Downloads\Topolino 3704\Topolino N.3704 - 18 Novembre 2026\edi011.jpg"
    print(f"Analyse de la planche : {test_path}...\n")
    extractor = ComicDialogueSpeakerExtractor(lang="it")
    res = extractor.process_page(test_path)
    print(format_as_screenplay(res))
