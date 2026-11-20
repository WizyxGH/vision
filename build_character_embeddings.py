"""
Extraction des embeddings DINOv2 sur les portraits de référence de personnages Disney
et construction de l'index vectoriel.
"""

import os
import json
import glob
import argparse
from typing import List, Dict
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

DEFAULT_REF_DIR = os.path.join(os.path.dirname(__file__), "data", "references")
DEFAULT_OUTPUT_INDEX = os.path.join(os.path.dirname(__file__), "data", "character_embeddings.npz")


def get_image_transform():
    """Transform standardisé sans dépendance stricte à torchvision."""
    def transform(img: Image.Image) -> torch.Tensor:
        # Convertir en RGB si nécessaire (ex: RGBA ou N&B)
        if img.mode != "RGB":
            # Gérer le fond transparent en blanc
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")

        # Redimensionnement 224x224 bicubic
        img = img.resize((224, 224), Image.Resampling.BICUBIC)

        # Normalisation standard ImageNet
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        # Transposer en [C, H, W]
        arr = arr.transpose((2, 0, 1))
        return torch.from_numpy(arr)

    return transform


def load_dinov2_model(device: str = "cpu"):
    """Charge le modèle DINOv2 ViT-S/14 (384 dimensions, rapide et léger)."""
    print(f"Chargement du modèle DINOv2 (dinov2_vits14) sur {device}...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    model.to(device)
    return model


def fix_mojibake(text: str) -> str:
    """Corrige les noms de personnages mal décodés (ex: DÃ³-RÃ©-Mi -> Dó-Ré-Mi)."""
    if not text:
        return ""
    if "Ã" in text or "Â" in text:
        try:
            return text.encode("latin-1").decode("utf-8")
        except Exception:
            pass
    return text


def scan_reference_images(ref_dir: str) -> List[Dict[str, str]]:
    """Parcourt les sous-dossiers par personnage pour lister toutes les images."""
    items = []
    if not os.path.exists(ref_dir):
        return items

    for hero_folder in os.listdir(ref_dir):
        folder_path = os.path.join(ref_dir, hero_folder)
        if not os.path.isdir(folder_path):
            continue
        hero = fix_mojibake(hero_folder)
        for file in os.listdir(folder_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                items.append({
                    "hero": hero,
                    "filename": file,
                    "filepath": os.path.join(folder_path, file)
                })
    return items


def extract_embeddings(items: List[Dict[str, str]], model, device: str = "cpu", batch_size: int = 16):
    """Calcule les vecteurs normalisés L2 par batch."""
    transform = get_image_transform()
    embeddings_list = []
    metadata_list = []

    for i in tqdm(range(0, len(items), batch_size), desc="Extraction embeddings"):
        batch_items = items[i:i + batch_size]
        tensors = []
        valid_meta = []

        for item in batch_items:
            try:
                with Image.open(item["filepath"]) as img:
                    tensor = transform(img)
                    tensors.append(tensor)
                    valid_meta.append({
                        "hero": item["hero"],
                        "filename": item["filename"],
                        "filepath": os.path.relpath(item["filepath"], os.path.dirname(__file__))
                    })
            except Exception as e:
                print(f"\nErreur lecture image {item['filepath']}: {e}")

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            out = model(batch_tensor)
            # Normalisation L2 (norme = 1.0) pour que le produit scalaire == similarité cosinus
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            emb = out.cpu().numpy()

        embeddings_list.append(emb)
        metadata_list.extend(valid_meta)

    if embeddings_list:
        all_embeddings = np.vstack(embeddings_list)
    else:
        all_embeddings = np.empty((0, 384), dtype=np.float32)

    return all_embeddings, metadata_list


def sanitize_and_build_prototypes(embeddings: np.ndarray, metadata: List[Dict[str, str]], anchor_codes: List[str] = None):
    """
    Filtre automatiquement les images de référence contaminées (ex: un personnage secondaire
    dont l'image Inducks montre en fait Donald ou Mickey en arrière-plan) et calcule le
    vecteur prototype (centroïde L2) de chaque personnage.
    """
    if anchor_codes is None:
        anchor_codes = ['DD', 'MM', 'US', 'GO', 'PL', 'DA', 'FE', 'HDL', 'GY', 'GL', 'PK', 'BB', 'MI']

    by_hero = {}
    for i, m in enumerate(metadata):
        by_hero.setdefault(m["hero"], []).append(i)

    # 1. Calcul des vecteurs des ancres majeures
    anchors = {}
    for code in anchor_codes:
        if code in by_hero:
            vec = np.mean(embeddings[by_hero[code]], axis=0)
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                anchors[code] = vec / norm

    # 2. Filtrage des contaminations et des outliers
    keep_indices = []
    removed_contamination = 0
    removed_outliers = 0

    for h, idxs in by_hero.items():
        h_embs = embeddings[idxs]
        h_centroid = np.mean(h_embs, axis=0)
        norm = np.linalg.norm(h_centroid)
        if norm > 1e-6:
            h_centroid /= norm

        clean_h_idxs = []
        for idx in idxs:
            vec = embeddings[idx]
            sim_to_self = float(np.dot(vec, h_centroid))

            # Test de contamination flagrante par une ancre majeure (Donald, Mickey, etc.)
            is_contaminated = False
            if h not in anchors and len(idxs) >= 2:
                for a_code, a_vec in anchors.items():
                    sim_to_a = float(np.dot(vec, a_vec))
                    # Rejette uniquement si l'image est indiscutablement le héros ancre (> 0.88)
                    # et très nettement plus proche de l'ancre que de son propre personnage
                    if sim_to_a > 0.88 and sim_to_a > sim_to_self + 0.15:
                        is_contaminated = True
                        break

            if is_contaminated:
                removed_contamination += 1
                continue

            # Note : Aucun rejet d'outlier stylistique (afin de préserver 100% des styles,
            # costumes et époques historiques légitimes).
            clean_h_idxs.append(idx)

        if not clean_h_idxs:
            clean_h_idxs = [idxs[0]]

        keep_indices.extend(clean_h_idxs)

    keep_indices = sorted(list(set(keep_indices)))
    clean_embs = embeddings[keep_indices]
    clean_meta = [metadata[i] for i in keep_indices]

    print(f"Assainissement : {len(clean_embs)}/{len(embeddings)} images retenues (-{removed_contamination} contaminées écartées, styles et époques 100% préservés)")

    # 3. Calcul des prototypes (centroïdes L2) par personnage
    proto_by_hero = {}
    for i, m in enumerate(clean_meta):
        proto_by_hero.setdefault(m["hero"], []).append(i)

    proto_heroes = sorted(list(proto_by_hero.keys()))
    prototypes = []
    for h in proto_heroes:
        h_idxs = proto_by_hero[h]
        h_vecs = clean_embs[h_idxs]
        c = np.mean(h_vecs, axis=0)
        norm = np.linalg.norm(c)
        if norm > 1e-6:
            c /= norm
        prototypes.append(c)

    prototypes = np.array(prototypes, dtype=np.float32)
    return clean_embs, clean_meta, prototypes, proto_heroes


def main():
    parser = argparse.ArgumentParser(description="Construit l'index d'embeddings DINOv2 des personnages Disney")
    parser.add_argument("--ref-dir", default=DEFAULT_REF_DIR, help="Dossier contenant les images de référence")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_INDEX, help="Fichier .npz de sortie pour l'index")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille des batchs d'inférence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu ou cuda)")

    args = parser.parse_args()

    print(f"Recherche des images de référence dans {args.ref_dir}...")
    items = scan_reference_images(args.ref_dir)
    print(f"Nombre total d'images trouvées : {len(items)}")

    if not items:
        print("Aucune image trouvée. Exécutez d'abord download_character_references.py.")
        return

    heroes = set(it["hero"] for it in items)
    print(f"Nombre de personnages représentés : {len(heroes)}")

    model = load_dinov2_model(device=args.device)

    print("Début du calcul des embeddings...")
    raw_embeddings, raw_metadata = extract_embeddings(items, model, device=args.device, batch_size=args.batch_size)

    print("Assainissement des données et calcul des prototypes (centroïdes)...")
    embeddings, metadata, prototypes, proto_heroes = sanitize_and_build_prototypes(raw_embeddings, raw_metadata)

    # Sauvegarde dans un fichier .npz compact et rapide
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta_json = json.dumps(metadata, ensure_ascii=False)
    proto_heroes_json = json.dumps(proto_heroes, ensure_ascii=False)

    np.savez_compressed(
        args.out,
        embeddings=embeddings,
        metadata=meta_json,
        prototypes=prototypes,
        prototype_heroes=proto_heroes_json
    )

    print(f"\n--- Index Vectoriel Généré avec Succès ---")
    print(f"Fichier : {args.out}")
    print(f"Dimensions de la matrice : {embeddings.shape} (Type: {embeddings.dtype})")
    print(f"Taille du fichier : {os.path.getsize(args.out) / (1024 * 1024):.2f} Mo")


if __name__ == "__main__":
    main()
