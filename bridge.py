"""
Passerelle d'intégration (Bridge) entre le moteur de vision apps/vision et dbi-generator.
Permet d'enrichir ou de déduire automatiquement le héros et la liste des apparitions (xapp)
lorsqu'une histoire de bande dessinée Disney n'est pas identifiée par son code officiel Inducks.
"""

import os
import sys
from typing import List, Tuple, Dict, Optional
from PIL import Image

from multimodal_character_fusion import MultiModalCharacterFusion, DEFAULT_INDEX_PATH
from visualizer import draw_detection_annotations

_FUSION_SINGLETON: Optional[MultiModalCharacterFusion] = None


def get_fusion_engine() -> MultiModalCharacterFusion:
    global _FUSION_SINGLETON
    if _FUSION_SINGLETON is None:
        _FUSION_SINGLETON = MultiModalCharacterFusion()
    return _FUSION_SINGLETON


def detect_characters_from_archive_story(
    archive,
    start_page: int,
    num_pages: int,
    current_hero: str = "",
    min_score: float = 0.70,
    lang: str = "it"
) -> Tuple[str, List[str], Dict[str, any]]:
    """
    Scanne les pages d'une histoire à partir d'une archive (.cbz, .cbr, .pdf ou LazyArchive).
    """
    fusion = get_fusion_engine()
    page_images: List[Image.Image] = []
    end_page = start_page + num_pages

    for p in range(start_page, end_page):
        try:
            if hasattr(archive, "get_page_image"):
                pil_img = archive.get_page_image(p)
            elif hasattr(archive, "get_page_bytes"):
                import io
                b = archive.get_page_bytes(p)
                pil_img = Image.open(io.BytesIO(b)).convert("RGB")
            else:
                continue
            if pil_img:
                page_images.append(pil_img)
        except Exception as e:
            print(f"  [Vision] Erreur lecture page {p}: {e}")

    if not page_images:
        return current_hero or "--", [], {}

    hero, xapp_sorted, details = fusion.analyze_story(
        page_images, lang=lang, min_score=min_score
    )

    if current_hero and current_hero != "--":
        hero = current_hero
        if hero not in xapp_sorted:
            xapp_sorted = sorted(xapp_sorted + [hero])
            details["xapp_str"] = ",".join(xapp_sorted)
            details["xapp_tag"] = f"[xapp:{details['xapp_str']}]"

    return hero, xapp_sorted, details


def process_single_target(target_path: str, fusion: MultiModalCharacterFusion, visualize: bool = True) -> Dict:
    """Traite une image ou archive unique et affiche les résultats détaillés."""
    ext = os.path.splitext(target_path)[1].lower()
    base_name = os.path.basename(target_path)
    stem_name = os.path.splitext(base_name)[0]

    print("\n" + "=" * 70)
    print(f"  📄 ANALYSE : {base_name}")
    print("=" * 70)

    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        img = Image.open(target_path).convert("RGB")
        hero, xapp, stats = fusion.analyze_story([img], min_score=0.70)

        # Génération de l'image annotée si demandée
        annotated_path = None
        if visualize:
            try:
                # Tenter de sauvegarder dans le même dossier
                parent_dir = os.path.dirname(os.path.abspath(target_path))
                out_file = os.path.join(parent_dir, f"{stem_name}_detection.jpg")
                annotated_path = draw_detection_annotations(
                    img=img,
                    title_hero=hero,
                    title_text=stats.get("title_text", ""),
                    panels=stats.get("panels", []),
                    character_detections=stats.get("all_detections", []),
                    output_path=out_file
                )
            except Exception:
                # Secours : dossier debug local
                debug_dir = os.path.join(os.path.dirname(__file__), "debug")
                out_file = os.path.join(debug_dir, f"{stem_name}_detection.jpg")
                annotated_path = draw_detection_annotations(
                    img=img,
                    title_hero=hero,
                    title_text=stats.get("title_text", ""),
                    panels=stats.get("panels", []),
                    character_detections=stats.get("all_detections", []),
                    output_path=out_file
                )

        # Affichage détaillé par case et zone
        panels = stats.get("panels", [])
        all_dets = stats.get("all_detections", [])
        print(f"\n🔍 DÉCOUPAGE : {len(panels)} case(s) identifiée(s)")
        for idx, (px1, py1, px2, py2) in enumerate(panels, start=1):
            pw, ph = px2 - px1, py2 - py1
            chars_in_p = [d for d in all_dets if d.get("panel_index") == idx]
            print(f"  ┌ Case {idx} : [x:{px1}, y:{py1}, w:{pw}, h:{ph}]")
            if chars_in_p:
                for c in chars_in_p:
                    h_code = c["hero"]
                    score = c.get("adjusted_score", c["score"]) * 100
                    bx = c["box"]
                    print(f"  │   • {h_code:<6} {score:5.1f}%  zone: [x:{bx[0]}, y:{bx[1]}, w:{bx[2]-bx[0]}, h:{bx[3]-bx[1]}]")
            else:
                print("  │   (aucun personnage identifié au seuil de confiance)")
            print("  └" + "─" * 45)

        # Mentions textuelles OCR
        if stats.get("ocr_counts"):
            print("\n💬 MENTIONS DIALOGUES / TEXTES (OCR) :")
            for h, count in stats["ocr_counts"].items():
                print(f"  • {h:<6} : {count} citation(s)")

        # Résultats Bolderbast
        print("\n" + "-" * 70)
        print("                      RÉSULTATS BOLDERBAST                      ")
        print("-" * 70)
        print(f"  Colonne hero (Titre / Logo) : {hero}")
        print(f"  Balise xapp (Sans espace)   : {stats['xapp_tag']}")
        print(f"  Univers dominant            : {stats.get('dominant_universe', 'Ducks')}")
        if annotated_path:
            print(f"  🖼️  Image annotée générée   : {annotated_path}")
        print("=" * 70)

        return {
            "target": base_name,
            "hero": hero,
            "xapp_tag": stats["xapp_tag"],
            "annotated_path": annotated_path
        }

    else:
        # Traitement d'une archive (.cbz, .cbr, .pdf)
        from pathlib import Path
        helper_path = Path(__file__).resolve().parents[1] / "Archiving-Helper" / "tools" / "dbi-generator"
        sys.path.insert(0, str(helper_path))
        from archive_loader import LazyArchive

        archive = LazyArchive(target_path)
        print(f"Archive chargée ({archive.total_pages} pages).")
        hero, xapp, stats = detect_characters_from_archive_story(
            archive, start_page=1, num_pages=min(4, archive.total_pages)
        )

        print("\n" + "-" * 70)
        print("                      RÉSULTATS BOLDERBAST                      ")
        print("-" * 70)
        print(f"  Colonne hero (Titre / Logo) : {hero}")
        print(f"  Balise xapp (Sans espace)   : {stats['xapp_tag']}")
        print(f"  Univers dominant            : {stats.get('dominant_universe', 'Ducks')}")
        print("=" * 70)

        return {
            "target": base_name,
            "hero": hero,
            "xapp_tag": stats["xapp_tag"],
            "annotated_path": None
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Disney Vision - Reconnaissance multi-modale de personnages (Bolderbast hero & xapp)"
    )
    parser.add_argument("targets", nargs="+", help="Chemin(s) vers une ou plusieurs images (.png, .jpg) ou archives (.cbz, .pdf)")
    parser.add_argument("--visualize", "-v", action="store_true", default=True, help="Générer une image annotée avec zones et boîtes")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize", help="Désactiver la génération d'images annotées")

    args = parser.parse_args()

    valid_targets = []
    for t in args.targets:
        cleaned = t.strip('"').strip("'")
        if os.path.exists(cleaned):
            valid_targets.append(cleaned)
        else:
            print(f"Fichier introuvable ignoré : {cleaned}")

    if not valid_targets:
        print("Aucun fichier valide à analyser.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("      DISNEY VISION - RECONNAISSANCE MULTI-MODALE DE PERSONNAGES      ")
    print("=" * 70)
    print(f"Nombre de fichier(s) à traiter : {len(valid_targets)}")

    fusion = get_fusion_engine()
    results = []

    for t in valid_targets:
        res = process_single_target(t, fusion=fusion, visualize=args.visualize)
        results.append(res)

    # Récapitulatif multi-fichiers si plusieurs fichiers ont été passés
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("                     RÉCAPITULATIF GLOBAL                        ")
        print("=" * 70)
        print(f"{'FICHIER':<30} | {'HERO':<6} | {'BALISE XAPP'}")
        print("-" * 70)
        for r in results:
            print(f"{r['target'][:30]:<30} | {r['hero']:<6} | {r['xapp_tag']}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
