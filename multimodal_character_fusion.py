"""
Moteur de Fusion Multi-Modale pour l'indexation des personnages Disney selon la norme Bolderbast.
Combine :
1. La détection visuelle de la tête/logo dans le bandeau de titre (Title Hero visuel).
2. L'OCR textuel du titre (Title Hero textuel).
3. La détection visuelle des personnages dans toutes les cases (DINOv2).
4. L'OCR des dialogues dans les bulles.
5. La cohérence du graphe d'univers Inducks (Ducks vs Mice vs Forest).
6. Les règles strictes de formatage Bolderbast (hero et [xapp:...]).
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Set
from PIL import Image

from page_character_detector import PageCharacterDetector, DEFAULT_INDEX_PATH
import re

# Univers Inducks principaux
UNIVERSE_DUCKS = {"DD", "US", "HDL", "DA", "FE", "PK", "GL", "GY", "BB", "MDS", "RKD", "LVD", "GD", "GU", "Battista", "Lusky", "FZ"}
UNIVERSE_MICE = {"MM", "GO", "PL", "MI", "CL", "HH", "PE", "PB", "EB", "IP", "CAS", "OH", "TR", "RS"}
UNIVERSE_FOREST = {"BRB", "BW", "JW", "Fifer Pig", "Fiddler Pig", "Practical Pig"}

# Dictionnaire OCR multi-langues des noms de personnages dans les dialogues
DIALOGUE_PATTERNS: Dict[str, List[Tuple[re.Pattern, str]]] = {
    "it": [
        (re.compile(r"\b(?:qui\s*,\s*quo\s*(?:,\s*|e\s*)qua|i\s+nipotini)\b", re.I), "HDL"),
        (re.compile(r"\b(?:banda\s+bassotti|i\s+bassotti|bassotti)\b", re.I), "BB"),
        (re.compile(r"\b(?:zio\s+paperone|paperone)\b", re.I), "US"),
        (re.compile(r"\b(?:macchia\s+nera)\b", re.I), "PB"),
        (re.compile(r"\b(?:pico\s+de\s+paperis)\b", re.I), "LVD"),
        (re.compile(r"\b(?:indiana\s+pipps)\b", re.I), "IP"),
        (re.compile(r"\b(?:nonna\s+papera)\b", re.I), "GD"),
        (re.compile(r"\b(?:commissario\s+basettoni)\b", re.I), "OH"),
        (re.compile(r"\b(?:archimede)\b", re.I), "GY"),
        (re.compile(r"\b(?:paperino)\b", re.I), "DD"),
        (re.compile(r"\b(?:paperina)\b", re.I), "DA"),
        (re.compile(r"\b(?:paperoga)\b", re.I), "FE"),
        (re.compile(r"\b(?:paperinik|pikappa)\b", re.I), "PK"),
        (re.compile(r"\b(?:gastone)\b", re.I), "GL"),
        (re.compile(r"\b(?:topolino)\b", re.I), "MM"),
        (re.compile(r"\b(?:pippo)\b", re.I), "GO"),
        (re.compile(r"\b(?:pluto)\b", re.I), "PL"),
        (re.compile(r"\b(?:minni|minnie)\b", re.I), "MI"),
        (re.compile(r"\b(?:gambadilegno)\b", re.I), "PE"),
        (re.compile(r"\b(?:amelia)\b", re.I), "MDS"),
        (re.compile(r"\b(?:rockerduck)\b", re.I), "RKD"),
        (re.compile(r"\b(?:battista)\b", re.I), "Battista"),
    ],
    "fr": [
        (re.compile(r"\b(?:riri\s*,\s*fifi\s*(?:,\s*|et\s*)loulou|riri\s+fifi\s+loulou|les\s+neveux)\b", re.I), "HDL"),
        (re.compile(r"\b(?:les\s+rapetou|rapetou)\b", re.I), "BB"),
        (re.compile(r"\b(?:oncle\s+picsou|picsou)\b", re.I), "US"),
        (re.compile(r"\b(?:fant[oô]me\s+noir)\b", re.I), "PB"),
        (re.compile(r"\b(?:g[eé]o\s+trouvetou)\b", re.I), "GY"),
        (re.compile(r"\b(?:pat\s+hibulaire)\b", re.I), "PE"),
        (re.compile(r"\b(?:miss\s+tick)\b", re.I), "MDS"),
        (re.compile(r"\b(?:flairsou)\b", re.I), "RKD"),
        (re.compile(r"\b(?:donald)\b", re.I), "DD"),
        (re.compile(r"\b(?:daisy)\b", re.I), "DA"),
        (re.compile(r"\b(?:popop)\b", re.I), "FE"),
        (re.compile(r"\b(?:fantomiald)\b", re.I), "PK"),
        (re.compile(r"\b(?:gontran)\b", re.I), "GL"),
        (re.compile(r"\b(?:mickey)\b", re.I), "MM"),
        (re.compile(r"\b(?:dingo)\b", re.I), "GO"),
        (re.compile(r"\b(?:pluto)\b", re.I), "PL"),
        (re.compile(r"\b(?:minnie)\b", re.I), "MI"),
    ],
    "en": [
        (re.compile(r"\b(?:huey\s*,\s*dewey\s*(?:,\s*|and\s*)louie|nephews)\b", re.I), "HDL"),
        (re.compile(r"\b(?:beagle\s+boys)\b", re.I), "BB"),
        (re.compile(r"\b(?:uncle\s+scrooge|scrooge)\b", re.I), "US"),
        (re.compile(r"\b(?:donald\s+duck|donald)\b", re.I), "DD"),
        (re.compile(r"\b(?:daisy\s+duck|daisy)\b", re.I), "DA"),
        (re.compile(r"\b(?:fethry\s+duck|fethry)\b", re.I), "FE"),
        (re.compile(r"\b(?:gyro\s+gearloose|gyro)\b", re.I), "GY"),
        (re.compile(r"\b(?:gladstone\s+gander|gladstone)\b", re.I), "GL"),
        (re.compile(r"\b(?:mickey\s+mouse|mickey)\b", re.I), "MM"),
        (re.compile(r"\b(?:goofy)\b", re.I), "GO"),
        (re.compile(r"\b(?:pluto)\b", re.I), "PL"),
        (re.compile(r"\b(?:minnie\s+mouse|minnie)\b", re.I), "MI"),
        (re.compile(r"\b(?:magica\s+de\s+spell|magica)\b", re.I), "MDS"),
    ]
}


class MultiModalCharacterFusion:
    def __init__(self, index_path: str = DEFAULT_INDEX_PATH):
        self.detector = PageCharacterDetector(index_path=index_path)

    def detect_title_hero(
        self,
        first_page_img: Image.Image,
        splash_title_text: str = "",
        lang: str = "it"
    ) -> str:
        """
        Détermine le Title Hero (colonne hero du .dbi) selon Bolderbast :
        1. Recherche d'un macaron/logo visuel avec la tête du personnage dans le bandeau de titre.
        2. Recherche d'un nom de personnage explicite dans le texte du titre.
        3. Si absent : retourne '--' conformément à Bolderbast.
        """
        pw, ph = first_page_img.size

        # A) Extraction du bandeau de titre (environ les 25-30% supérieurs de la page de splash)
        title_banner = first_page_img.crop((0, 0, pw, int(ph * 0.32)))
        banner_detections = self.detector.detect_in_panel(title_banner, min_score=0.76)

        if banner_detections:
            # On privilégie un personnage nettement identifié dans le bandeau du titre
            visual_title_hero = banner_detections[0]["hero"]
            return visual_title_hero

        # B) Analyse textuelle du titre imprimé (OCR)
        if splash_title_text:
            patterns = DIALOGUE_PATTERNS.get(lang.lower(), DIALOGUE_PATTERNS["it"])
            for pat, code in patterns:
                if pat.search(splash_title_text):
                    return code

        # C) Règle Bolderbast : aucun personnage dans le titre = '--'
        return "--"

    def analyze_story(
        self,
        page_images: List[Image.Image],
        dialogue_texts: Optional[Dict[int, List[str]]] = None,
        splash_title_text: str = "",
        lang: str = "it",
        min_score: float = 0.70
    ) -> Tuple[str, List[str], Dict[str, any]]:
        """
        Analyse complète d'une histoire par fusion multi-modale :
        Retourne (title_hero, [xapp_sorted], details)
        """
        num_pages = len(page_images)
        if num_pages == 0:
            return "--", [], {}

        # 1. Détermination du Title Hero sur la première page (Bandeau de titre)
        title_hero = self.detect_title_hero(page_images[0], splash_title_text=splash_title_text, lang=lang)

        # 2. Analyse visuelle de toutes les cases de l'histoire
        visual_appearances: Dict[str, int] = {}
        visual_max_scores: Dict[str, float] = {}

        for p_idx, page_img in enumerate(page_images, start=1):
            page_res = self.detector.analyze_page(page_img, min_score=min_score)
            for h, cnt in page_res["hero_counts"].items():
                visual_appearances[h] = visual_appearances.get(h, 0) + cnt
                visual_max_scores[h] = max(visual_max_scores.get(h, 0.0), page_res["hero_max_scores"][h])

        # 3. Analyse OCR des dialogues
        ocr_counts: Dict[str, int] = {}
        if dialogue_texts:
            patterns = DIALOGUE_PATTERNS.get(lang.lower(), DIALOGUE_PATTERNS["it"])
            all_dialogue = " ".join(
                line for lines in dialogue_texts.values() for line in lines
            )
            for pat, code in patterns:
                matches = len(pat.findall(all_dialogue))
                if matches > 0:
                    ocr_counts[code] = matches

        # 4. Détermination de l'univers dominant (Prior d'univers Inducks)
        ducks_votes = sum(visual_appearances.get(h, 0) for h in UNIVERSE_DUCKS)
        mice_votes = sum(visual_appearances.get(h, 0) for h in UNIVERSE_MICE)
        forest_votes = sum(visual_appearances.get(h, 0) for h in UNIVERSE_FOREST)

        dominant_universe = "Ducks"
        if mice_votes > ducks_votes and mice_votes > forest_votes:
            dominant_universe = "Mice"
        elif forest_votes > ducks_votes and forest_votes > mice_votes:
            dominant_universe = "Forest"

        # 5. Fusion et validation de la liste XAPP
        # Seuil minimal d'apparitions : 1 case pour les gags 1-2 pages, 2 cases sinon
        min_cases = 1 if num_pages <= 2 else 2
        validated_xapp: Set[str] = set()

        all_candidates = set(visual_appearances.keys()) | set(ocr_counts.keys())

        for cand in all_candidates:
            v_cnt = visual_appearances.get(cand, 0)
            v_score = visual_max_scores.get(cand, 0.0)
            t_cnt = ocr_counts.get(cand, 0)

            # Règle A : Vu visuellement sur >= min_cases
            if v_cnt >= min_cases and v_score >= min_score:
                validated_xapp.add(cand)

            # Règle B : Vu visuellement (score >= 0.70) ET mentionné dans les dialogues
            elif v_cnt >= 1 and t_cnt >= 1:
                validated_xapp.add(cand)

            # Règle C : Score visuel très fort (>= 0.84) même sur 1 case
            elif v_cnt >= 1 and v_score >= 0.84:
                validated_xapp.add(cand)

            # Règle D : Mentionné plusieurs fois dans les dialogues (>= 2) dans l'univers dominant
            elif t_cnt >= 2:
                if (dominant_universe == "Ducks" and cand in UNIVERSE_DUCKS) or \
                   (dominant_universe == "Mice" and cand in UNIVERSE_MICE):
                    validated_xapp.add(cand)

        # Règle Dizni #210 : si un Title Hero est défini (différent de '--'), il DOIT être dans xapp
        if title_hero and title_hero != "--":
            validated_xapp.add(title_hero)

        # Tri alphabétique standard Bolderbast
        xapp_sorted = sorted(list(validated_xapp))
        xapp_str = ",".join(xapp_sorted)  # Strictement sans espace : DD,HDL,US

        details = {
            "title_hero": title_hero,
            "xapp_str": xapp_str,
            "xapp_tag": f"[xapp:{xapp_str}]" if xapp_str else "",
            "dominant_universe": dominant_universe,
            "visual_appearances": visual_appearances,
            "visual_max_scores": visual_max_scores,
            "ocr_counts": ocr_counts,
            "total_pages": num_pages
        }

        return title_hero, xapp_sorted, details
