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
                b = archive.get_page_bytes(p)
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
                p1_img = Image.open(io.BytesIO(archive.get_page_bytes(start_page))).convert("RGB")
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
