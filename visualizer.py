"""
Module de visualisation et de débogage visuel des détections Disney Vision.
Génère une image annotée avec :
- Le bandeau de titre en jaune (avec le héros de titre et texte OCR extrait)
- Les cases segmentées en cyan (avec numéro de case et dimensions)
- Les boîtes englobantes des personnages détectés en vert (avec code héros et score)
"""

import os
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


def draw_detection_annotations(
    img: Image.Image,
    title_hero: str,
    title_text: str,
    panels: List[Tuple[int, int, int, int]],
    character_detections: List[Dict],
    output_path: str,
    title_banner_ratio: float = 0.30
) -> str:
    """
    Génère et sauvegarde une image haute résolution annotée avec toutes les zones détectées.
    """
    annotated = img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    # Choix de police ou police par défaut
    try:
        font_large = ImageFont.truetype("arial.ttf", max(18, int(w * 0.022)))
        font_med = ImageFont.truetype("arial.ttf", max(14, int(w * 0.016)))
        font_small = ImageFont.truetype("arial.ttf", max(11, int(w * 0.012)))
    except Exception:
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_small = ImageFont.load_default()

    line_w = max(3, int(w * 0.003))

    # 1. Dessiner le bandeau de titre supérieur (Jaune / Or)
    if title_hero and title_hero != "--":
        b_h = int(h * title_banner_ratio)
        draw.rectangle([0, 0, w - 1, b_h], outline=(255, 215, 0), width=line_w)
        banner_label = f"★ TITRE HERO: {title_hero} ({title_text})" if title_text else f"★ TITRE HERO: {title_hero}"
        draw.rectangle([0, 0, min(w, int(len(banner_label) * 14)), max(28, int(h * 0.025))], fill=(255, 215, 0))
        draw.text((8, 4), banner_label, fill=(0, 0, 0), font=font_large)

    # 2. Dessiner chaque case segmentée (Cyan / Bleu)
    for idx, (x1, y1, x2, y2) in enumerate(panels, start=1):
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 255), width=line_w)
        panel_label = f"Case {idx} [{x2-x1}x{y2-y1}]"
        lbl_w = len(panel_label) * 9 + 10
        lbl_h = max(20, int(h * 0.018))
        draw.rectangle([x1, y1, min(x2, x1 + lbl_w), y1 + lbl_h], fill=(0, 180, 230))
        draw.text((x1 + 4, y1 + 2), panel_label, fill=(255, 255, 255), font=font_small)

    # 3. Dessiner chaque personnage détecté (Vert Lime)
    for det in character_detections:
        box = det.get("box")
        if not box:
            continue
        bx1, by1, bx2, by2 = box
        hero = det.get("hero", "???")
        score = det.get("score", 0.0) * 100

        draw.rectangle([bx1, by1, bx2, by2], outline=(50, 230, 50), width=line_w + 1)
        tag_label = f" {hero} ({score:.1f}%) "
        tag_w = len(tag_label) * 11 + 8
        tag_h = max(24, int(h * 0.022))

        # Positionner l'étiquette au-dessus ou à l'intérieur si trop haut
        ty1 = max(0, by1 - tag_h) if by1 >= tag_h else by1
        draw.rectangle([bx1, ty1, bx1 + tag_w, ty1 + tag_h], fill=(30, 180, 30))
        draw.text((bx1 + 4, ty1 + 2), tag_label, fill=(255, 255, 255), font=font_med)

    # Création du dossier parent si besoin
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    annotated.save(output_path, quality=92)
    return output_path
