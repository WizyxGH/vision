"""
Script de téléchargement des portraits et vignettes de référence de personnages Disney
depuis le catalogue officiel Inducks (characterscans.dbp).

Permet de constituer le jeu de données pour l'extraction d'embeddings DINOv2.
"""

import os
import re
import sys
import time
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from tqdm import tqdm

DEFAULT_DBP_PATH = r"C:\Users\starl\Documents\Projets\BD\Inducks\webinput\characterscans.dbp"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "references")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"


def sanitize_filename(name: str) -> str:
    """Remplace les caractères interdits pour un dossier ou fichier Windows."""
    return re.sub(r'[\\/*?:"<>| ]', '_', name)


def parse_characterscans(dbp_path: str) -> Dict[str, List[dict]]:
    """
    Lit characterscans.dbp et retourne un dictionnaire {hero: [records]}.
    Format de ligne: hero^country^entrycode^legend^login^pagenr^submitdate^url
    """
    if not os.path.exists(dbp_path):
        raise FileNotFoundError(f"Fichier de scan Inducks introuvable: {dbp_path}")

    characters: Dict[str, List[dict]] = {}
    with open(dbp_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("^")
            if len(parts) >= 8:
                hero, country, entrycode, legend, login, pagenr, submitdate, url = parts[:8]
                if not hero or not url:
                    continue
                hero_clean = hero.strip()
                if hero_clean not in characters:
                    characters[hero_clean] = []
                characters[hero_clean].append({
                    "hero": hero_clean,
                    "country": country,
                    "entrycode": entrycode,
                    "legend": legend,
                    "login": login,
                    "pagenr": pagenr,
                    "submitdate": submitdate,
                    "url": url.strip()
                })
    return characters


def download_single_image(record: dict, output_dir: str, retries: int = 2) -> Tuple[bool, str]:
    """Télécharge une image via le proxy inducks.org/hr.php."""
    hero = record["hero"]
    url = record["url"]
    hero_dir = os.path.join(output_dir, sanitize_filename(hero))
    os.makedirs(hero_dir, exist_ok=True)

    img_name = os.path.basename(url.split("?")[0])
    if not img_name:
        img_name = f"{record['country']}_{sanitize_filename(record['entrycode'])}_{record['pagenr']}.png"

    dest_path = os.path.join(hero_dir, img_name)

    # Si déjà présent et non vide, on ignore
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 500:
        return True, "cached"

    download_url = f"https://inducks.org/hr.php?normalsize=1&image={url}"
    req = urllib.request.Request(download_url, headers={"User-Agent": USER_AGENT})

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
                # Vérification que ce n'est pas un message d'erreur texte
                if len(data) > 500:
                    with open(dest_path, "wb") as f_out:
                        f_out.write(data)
                    return True, "downloaded"
                else:
                    return False, "too_small"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, "404"
            if attempt == retries:
                return False, f"http_{e.code}"
            time.sleep(0.5)
        except Exception as e:
            if attempt == retries:
                return False, str(e)
            time.sleep(0.5)

    return False, "failed"


def main():
    parser = argparse.ArgumentParser(description="Télécharge les vignettes de référence des personnages Disney Inducks")
    parser.add_argument("--dbp", default=DEFAULT_DBP_PATH, help="Chemin vers characterscans.dbp")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="Dossier de destination")
    parser.add_argument("--max-per-char", type=int, default=10, help="Nombre max de scans par personnage (0 pour illimité, défaut 10)")
    parser.add_argument("--min-scans", type=int, default=2, help="Filtrer les personnages ayant au moins N scans (défaut 2)")
    parser.add_argument("--characters", type=str, default="", help="Liste de codes spécifiques séparés par virgules (ex: DD,US,MM,FE,GO,PK)")
    parser.add_argument("--workers", type=int, default=8, help="Nombre de threads parallèles pour le téléchargement")
    parser.add_argument("--top-chars", type=int, default=0, help="Prendre seulement les Top N personnages les plus fréquents (0 pour tous)")

    args = parser.parse_args()

    print(f"Chargement de {args.dbp}...")
    chars_dict = parse_characterscans(args.dbp)
    print(f"Total de personnages trouvés dans le catalogue: {len(chars_dict)}")

    # Filtrage
    selected_heroes = list(chars_dict.keys())

    if args.characters:
        target_set = set(c.strip() for c in args.characters.split(","))
        selected_heroes = [h for h in selected_heroes if h in target_set]
        print(f"Filtré sur {len(selected_heroes)} personnages spécifiés: {', '.join(selected_heroes)}")
    else:
        # Tri par nombre de scans décroissant
        selected_heroes.sort(key=lambda h: len(chars_dict[h]), reverse=True)

        if args.min_scans > 1:
            selected_heroes = [h for h in selected_heroes if len(chars_dict[h]) >= args.min_scans]
            print(f"Personnages avec au moins {args.min_scans} scans: {len(selected_heroes)}")

        if args.top_chars > 0:
            selected_heroes = selected_heroes[:args.top_chars]
            print(f"Limité au Top {args.top_chars} personnages.")

    # Préparation de la liste des téléchargements
    tasks = []
    for h in selected_heroes:
        records = chars_dict[h]
        if args.max_per_char > 0:
            records = records[:args.max_per_char]
        for rec in records:
            tasks.append(rec)

    print(f"\nTotal d'images à télécharger/vérifier: {len(tasks)} (pour {len(selected_heroes)} personnages)")
    print(f"Dossier de sortie: {args.out}")
    os.makedirs(args.out, exist_ok=True)

    success_cnt = 0
    cached_cnt = 0
    fail_cnt = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_single_image, rec, args.out): rec for rec in tasks}
        with tqdm(total=len(tasks), desc="Téléchargement", unit="img") as pbar:
            for fut in as_completed(futures):
                ok, status = fut.result()
                if ok:
                    if status == "cached":
                        cached_cnt += 1
                    else:
                        success_cnt += 1
                else:
                    fail_cnt += 1
                pbar.update(1)

    print("\n--- Bilan du Téléchargement ---")
    print(f"Nouveaux téléchargements réussis : {success_cnt}")
    print(f"Déjà en cache                  : {cached_cnt}")
    print(f"Échecs                         : {fail_cnt}")
    print(f"Images prêtes dans             : {args.out}")


if __name__ == "__main__":
    main()
