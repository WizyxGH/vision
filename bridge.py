"""
Passerelle d'intégration (Bridge) entre le moteur de vision apps/vision et dbi-generator.
Permet d'enrichir ou de déduire automatiquement le héros et la liste des apparitions (xapp)
lorsqu'une histoire de bande dessinée Disney n'est pas identifiée par son code officiel Inducks.
"""

import os
from typing import List, Tuple, Dict, Optional
from PIL import Image

from page_character_detector import PageCharacterDetector, DEFAULT_INDEX_PATH

_DETECTOR_SINGLETON: Optional[PageCharacterDetector] = None


def get_detector() -> PageCharacterDetector:
    global _DETECTOR_SINGLETON
    if _DETECTOR_SINGLETON is None:
        _DETECTOR_SINGLETON = PageCharacterDetector()
    return _DETECTOR_SINGLETON


def detect_characters_from_archive_story(
    archive,
    start_page: int,
    num_pages: int,
    current_hero: str = "",
    min_score: float = 0.72,
    min_panel_appearances: int = 1
) -> Tuple[str, List[str], Dict[str, any]]:
    """
    Scanne les pages d'une histoire à partir d'une archive (.cbz, .cbr, .pdf ou ArchiveLoader).

    Args:
        archive: Objet archive (avec get_page_image ou get_page_bytes) ou chemin d'archive
        start_page: Index de la première page (1-based)
        num_pages: Nombre de pages de l'histoire
        current_hero: Héros déjà identifié (optionnel)
        min_score: Seuil de similarité cosinus DINOv2
        min_panel_appearances: Nombre minimal de cases pour valider une apparition

    Returns:
        (hero_code, [xapp_codes_tries], stats_dict)
    """
    detector = get_detector()
    hero_counts: Dict[str, int] = {}
    hero_scores: Dict[str, float] = {}

    end_page = start_page + num_pages

    for p in range(start_page, end_page):
        try:
            # Récupération de l'image de la page
            if hasattr(archive, "get_page_image"):
                pil_img = archive.get_page_image(p)
            elif hasattr(archive, "get_page_bytes"):
                import io
                # ArchiveLoader utilise des index 0-based, alors que start_page est 1-based
                idx = p - 1 if p >= 1 else p
                b = archive.get_page_bytes(idx)
                pil_img = Image.open(io.BytesIO(b)).convert("RGB")
            else:
                continue

            page_res = detector.analyze_page(pil_img, min_score=min_score)

            for h, count in page_res["hero_counts"].items():
                hero_counts[h] = hero_counts.get(h, 0) + count
                hero_scores[h] = max(hero_scores.get(h, 0.0), page_res["hero_max_scores"][h])

        except Exception as e:
            print(f"  [Vision] Erreur d'analyse page {p}: {e}")

    # Filtrer par persistance (au moins N cases)
    # Pour les histoires très courtes (1-2 pages), 1 apparition suffit.
    threshold = 1 if num_pages <= 2 else min_panel_appearances
    validated_heroes = [
        h for h, cnt in hero_counts.items()
        if cnt >= threshold
    ]

    # Déterminer le Title Hero (colonne hero Bolderbast)
    # Règle Bolderbast : le héros est UNIQUEMENT celui du titre/logo/macaron imprimé sur la page 1,
    # ou '--' s'il n'y a pas de héros dans le titre. Ce n'est JAMAIS le personnage le plus fréquent.
    hero = current_hero
    if not hero:
        # Tenter la détection visuelle du macaron/tête dans le bandeau supérieur de la page 1 (splash)
        try:
            if hasattr(archive, "get_page_image"):
                p1_img = archive.get_page_image(start_page)
            elif hasattr(archive, "get_page_bytes"):
                import io
                idx1 = start_page - 1 if start_page >= 1 else start_page
                p1_img = Image.open(io.BytesIO(archive.get_page_bytes(idx1))).convert("RGB")
            else:
                p1_img = None

            if p1_img:
                pw, ph = p1_img.size
                banner = p1_img.crop((0, 0, pw, int(ph * 0.32)))
                banner_detections = detector.detect_in_panel(banner, min_score=0.76)
                if banner_detections:
                    hero = banner_detections[0]["hero"]
        except Exception as e:
            print(f"  [Vision] Erreur détection macaron titre: {e}")

    # Si aucun héros n'est dans le titre/logo, la norme Bolderbast impose '--'
    if not hero:
        hero = "--"

    # Construire la liste xapp (alphabétique pour respecter Bolderbast)
    xapp = set(validated_heroes)
    # Dizni règle #210 : si un héros de titre est défini, il DOIT être dans xapp
    if hero and hero != "--":
        xapp.add(hero)

    xapp_sorted = sorted(list(xapp))
    xapp_str = ",".join(xapp_sorted)  # Strictement sans espace: DD,HDL,US
    xapp_tag = f"[xapp:{xapp_str}]" if xapp_str else ""

    stats = {
        "title_hero": hero,
        "xapp_str": xapp_str,
        "xapp_tag": xapp_tag,
        "hero_counts": hero_counts,
        "hero_scores": hero_scores,
        "scanned_pages": num_pages
    }

    return hero, xapp_sorted, stats


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Analyse visuelle et détection de personnages Disney (Bolderbast hero & xapp)")
    parser.add_argument("target", help="Chemin vers une image (.png, .jpg) ou une archive (.cbz, .cbr, .pdf)")
    parser.add_argument("--start", type=int, default=1, help="Page de début (pour une archive, défaut 1)")
    parser.add_argument("--pages", type=int, default=1, help="Nombre de pages de l'histoire (défaut 1)")
    parser.add_argument("--hero", type=str, default="", help="Héros déjà connu (optionnel)")

    args = parser.parse_args()

    if not os.path.exists(args.target):
        print(f"Fichier introuvable : {args.target}")
        sys.exit(1)

    ext = os.path.splitext(args.target)[1].lower()

    print("\n" + "=" * 65)
    print("      🎨 DISNEY VISION - DÉTECTION MULTI-MODALE DE PERSONNAGES      ")
    print("=" * 65)
    print(f"Fichier analysé : {os.path.basename(args.target)}")

    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        # Analyse directe d'une image unique (page ou gag)
        img = Image.open(args.target).convert("RGB")
        detector = get_detector()
        print("[1/2] Segmentation des cases et reconnaissance visuelle...")
        page_res = detector.analyze_page(img, min_score=0.72)

        # Détection du macaron dans le bandeau supérieur (splash)
        pw, ph = img.size
        banner = img.crop((0, 0, pw, int(ph * 0.32)))
        banner_detections = detector.detect_in_panel(banner, min_score=0.76)
        title_hero = banner_detections[0]["hero"] if banner_detections else (args.hero or "--")

        detected_chars = set(page_res["hero_counts"].keys())
        if title_hero and title_hero != "--":
            detected_chars.add(title_hero)

        xapp_sorted = sorted(list(detected_chars))
        xapp_str = ",".join(xapp_sorted)
        xapp_tag = f"[xapp:{xapp_str}]" if xapp_str else ""

        print(f"[2/2] Analyse terminée : {len(page_res['panels'])} case(s) détectée(s).")
        print("\n" + "-" * 65)
        print("                      RÉSULTATS BOLDERBAST                      ")
        print("-" * 65)
        print(f"  Colonne hero (Titre / Logo) : {title_hero}")
        print(f"  Balise xapp (Sans espace)   : {xapp_tag}")
        print("-" * 65)
        print("Détail des apparitions :")
        for h, cnt in page_res["hero_counts"].items():
            score = page_res["hero_max_scores"][h] * 100
            print(f"  • {h:<15} : {cnt} case(s) (similarité max : {score:.1f}%)")
        print("=" * 65 + "\n")

    else:
        # Analyse d'une archive
        import sys
        from pathlib import Path
        helper_path = Path(__file__).resolve().parents[1] / "Archiving-Helper" / "tools" / "dbi-generator"
        sys.path.insert(0, str(helper_path))
        from archive_loader import LazyArchive

        archive = LazyArchive(args.target)
        print(f"Archive chargée ({archive.total_pages} pages). Analyse des pages {args.start} à {args.start + args.pages - 1}...")

        hero, xapp, stats = detect_characters_from_archive_story(
            archive, start_page=args.start, num_pages=args.pages, current_hero=args.hero
        )

        print("\n" + "-" * 65)
        print("                      RÉSULTATS BOLDERBAST                      ")
        print("-" * 65)
        print(f"  Colonne hero (Titre / Logo) : {hero}")
        print(f"  Balise xapp (Sans espace)   : {stats['xapp_tag']}")
        print("-" * 65)
        print("Détail des apparitions :")
        for h, cnt in stats["hero_counts"].items():
            score = stats["hero_scores"][h] * 100
            print(f"  • {h:<15} : {cnt} case(s) (similarité max : {score:.1f}%)")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
