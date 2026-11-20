"""
Pipeline de détection et reconnaissance de personnages Disney sur une image de BD.
Combine la localisation (YOLOv8) et l'identification (DINOv2).
"""

import os
from PIL import Image
from ultralytics import YOLO
from search_character import CharacterMatcher

# Classes d'intérêt potentielles dans un modèle généraliste COCO
INTERESTING_CLASSES = {"person", "bird", "cat", "dog", "teddy bear", "horse", "sheep", "cow"}

def run_pipeline(image_path: str, min_conf: float = 0.25):
    if not os.path.exists(image_path):
        print(f"Image introuvable : {image_path}")
        return

    print(f"Chargement de l'image : {image_path}")
    im = Image.open(image_path)
    print(f"Dimensions : {im.width} x {im.height}")

    print("Détection des personnages avec YOLO...")
    detector = YOLO("yolov8n.pt")
    results = detector(image_path, conf=min_conf, verbose=False)

    boxes = []
    for r in results:
        for box in r.boxes:
            cls_name = detector.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cls_name in INTERESTING_CLASSES:
                xyxy = [int(x) for x in box.xyxy[0].tolist()]
                boxes.append((xyxy, cls_name, conf))

    print(f"Total de cibles candidates détectées : {len(boxes)}")
    if not boxes:
        print("Aucun personnage détecté par YOLO.")
        return

    matcher = CharacterMatcher()

    for idx, (box, cls_name, conf) in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        # Marge de 15% pour inclure le contexte de la tête/buste
        x1_m = max(0, int(x1 - 0.15 * bw))
        y1_m = max(0, int(y1 - 0.15 * bh))
        x2_m = min(im.width, int(x2 + 0.15 * bw))
        y2_m = min(im.height, int(y2 + 0.15 * bh))

        crop = im.crop((x1_m, y1_m, x2_m, y2_m))
        predictions = matcher.identify(crop, top_k=3, aggregate_by_hero=True)

        top1 = predictions[0]
        top1_pct = top1["score"] * 100
        print(f"\n[Cible #{idx}] Zone ({x1_m}, {y1_m}, {x2_m}, {y2_m}) | YOLO: {cls_name} ({conf:.2f})")
        print(f"  👉 Top 1 : {top1['hero']:<15} (Confiance: {top1_pct:5.1f}%)")
        for rank, p in enumerate(predictions[1:], start=2):
            print(f"     #{rank}   : {p['hero']:<15} ({p['score']*100:5.1f}%)")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\starl\Documents\Projets\Sites\DisneyComicsHub\panel1_crop.png"
    run_pipeline(target)
