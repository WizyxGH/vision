"""
Recherche et identification d'un personnage Disney dans l'index d'embeddings DINOv2.
Prend en entrée une image (visage ou buste découpé) et retourne les correspondances les plus proches.
"""

import os
import json
import argparse
from typing import List, Dict
import numpy as np
import torch
from PIL import Image

DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "character_embeddings.npz")


# Personnages majeurs de l'univers Disney (prior bayésien pour éviter les imposteurs de la longue traîne)
MAJOR_DISNEY_CHARACTERS = {
    "DD", "US", "HDL", "DA", "FE", "GL", "GY", "MM", "GO", "PL",
    "MI", "CL", "HH", "PE", "PB", "PK", "BB", "MDS", "RKD", "LVD",
    "GD", "GU", "Battista", "Lusky", "FZ", "IP", "CAS", "OH", "TR", "RS"
}


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


class CharacterMatcher:
    def __init__(self, index_path: str = DEFAULT_INDEX_PATH, device: str = None):
        self.index_path = index_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.embeddings = None
        self.metadata = None
        self.load_index()

    def load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index vectoriel introuvable: {self.index_path}. Lancez build_character_embeddings.py d'abord.")

        data = np.load(self.index_path, allow_pickle=True)
        self.embeddings = data["embeddings"]  # Shape: (N, 384)
        raw_meta = json.loads(str(data["metadata"]))
        for m in raw_meta:
            m["hero"] = fix_mojibake(m["hero"])
        self.metadata = raw_meta
        self.prototypes = data["prototypes"] if "prototypes" in data else None
        if "prototype_heroes" in data:
            self.prototype_heroes = [fix_mojibake(h) for h in json.loads(str(data["prototype_heroes"]))]
        else:
            self.prototype_heroes = None

        proto_info = f" + {len(self.prototype_heroes)} prototypes" if self.prototype_heroes else ""
        print(f"Index chargé : {len(self.metadata)} références vectorielles assainies{proto_info}.")

    def _init_model(self):
        if self.model is None:
            print(f"Chargement de DINOv2 sur {self.device}...")
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.model.eval()
            self.model.to(self.device)

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")

        img = img.resize((224, 224), Image.Resampling.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose((2, 0, 1))
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def identify(self, img_input, top_k: int = 5, aggregate_by_hero: bool = True) -> List[Dict]:
        """
        Identifie le personnage à partir d'un chemin d'image ou d'un objet PIL.Image.
        Retourne une liste des meilleurs candidats avec score de similarité cosinus (0.0 à 1.0).
        """
        self._init_model()

        if isinstance(img_input, str):
            with Image.open(img_input) as img:
                tensor = self._preprocess(img)
        else:
            tensor = self._preprocess(img_input)

        with torch.no_grad():
            query_emb = self.model(tensor)
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            query_vec = query_emb.cpu().numpy()[0]  # Shape: (384,)

        # Produit scalaire = similarité cosinus (car vecteurs déjà normalisés L2)
        similarities = np.dot(self.embeddings, query_vec)

        # Calcul des similarités avec les prototypes (centroïdes sans bruit)
        proto_scores = {}
        if self.prototypes is not None and self.prototype_heroes:
            p_sims = np.dot(self.prototypes, query_vec)
            for h, s in zip(self.prototype_heroes, p_sims):
                proto_scores[h] = float(s)

        if not aggregate_by_hero:
            # Top-K brut par image
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                meta = self.metadata[idx]
                results.append({
                    "hero": meta["hero"],
                    "score": float(similarities[idx]),
                    "reference_file": meta["filename"],
                    "reference_path": meta["filepath"]
                })
            return results

        # Agrégation par héros (avec régularisation par le prototype de personnage)
        hero_scores: Dict[str, dict] = {}
        for idx, score in enumerate(similarities):
            hero = self.metadata[idx]["hero"]
            if hero not in hero_scores or score > hero_scores[hero]["raw_score"]:
                p_score = proto_scores.get(hero, float(score))
                # Score équilibré : 70% meilleure référence, 30% centroïde
                # Protège contre les parasites qui n'ont pollué qu'une seule image
                combined_score = 0.7 * float(score) + 0.3 * p_score
                # Prior bayésien : léger bonus (+0.035) pour les personnages récurrents
                # évite qu'un personnage à 1 seule référence n'usurpe Donald, Mickey ou Picsou
                if hero in MAJOR_DISNEY_CHARACTERS:
                    combined_score += 0.035
                hero_scores[hero] = {
                    "hero": hero,
                    "score": combined_score,
                    "raw_score": float(score),
                    "proto_score": p_score,
                    "best_reference": self.metadata[idx]["filename"],
                    "filepath": self.metadata[idx]["filepath"]
                }

        sorted_heroes = sorted(hero_scores.values(), key=lambda x: x["score"], reverse=True)
        return sorted_heroes[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Identifie un personnage Disney à partir d'une image")
    parser.add_argument("image", help="Chemin vers l'image du visage/buste à identifier")
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help="Chemin de l'index vectoriel")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de prédictions à afficher")
    parser.add_argument("--raw", action="store_true", help="Afficher les correspondances brutes par image sans agréger par héros")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Fichier image introuvable: {args.image}")
        return

    matcher = CharacterMatcher(index_path=args.index)
    results = matcher.identify(args.image, top_k=args.top_k, aggregate_by_hero=not args.raw)

    print(f"\n--- Résultats d'Identification pour: {os.path.basename(args.image)} ---")
    for rank, res in enumerate(results, start=1):
        score_pct = res["score"] * 100
        ref = res.get("best_reference") or res.get("reference_file")
        print(f"  #{rank} [Score: {score_pct:5.1f}%]  Héros: {res['hero']:<15} (Réf: {ref})")


if __name__ == "__main__":
    main()
