"""
Découpeur automatique de cases (panels) pour planches de bandes dessinées Disney.
Utilise l'analyse des gouttières blanches/fond et la détection de contours rectangulaires.
"""

import os
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image


def extract_panels(image_path: str, min_panel_area_ratio: float = 0.02) -> List[Tuple[int, int, int, int]]:
    """
    Extrait les coordonnées [x, y, w, h] des cases d'une page de scan de BD.
    Retourne la liste des boîtes triées dans l'ordre de lecture (haut en bas, gauche à droite).
    """
    # Lecture avec PIL pour éviter les limitations de chunks OpenCV
    pil_img = Image.open(image_path).convert("RGB")
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h_page, w_page = gray.shape
    total_area = h_page * w_page

    # Binarisation pour faire ressortir les gouttières blanches et les bordures noires
    _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

    # Fermeture morphologique pour fusionner le contenu intérieur de chaque case
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Trouver les contours extérieurs
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panels = []
    min_area = total_area * min_panel_area_ratio
    max_area = total_area * 0.95

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Filtrer les bordures de page, les bandes trop fines ou les petits artefacts
        if min_area < area < max_area and w > w_page * 0.15 and h > h_page * 0.08:
            panels.append((x, y, w, h))

    # Tri dans l'ordre de lecture (par bandes horizontales d'abord, puis de gauche à droite)
    # On groupe les cases dont le Y est similaire (à 5% de la hauteur de la page près)
    def reading_order_key(box):
        x, y, w, h = box
        band = int(y / (h_page * 0.06))
        return (band, x)

    panels.sort(key=reading_order_key)
    return panels


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\starl\Documents\Projets\Sites\DisneyComicsHub\page3_preview.png"
    boxes = extract_panels(test_img)
    print(f"Cases détectées sur {os.path.basename(test_img)} : {len(boxes)}")
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        print(f"  Case #{i:2d} : x={x:4d}, y={y:4d}, w={w:4d}, h={h:4d}")
