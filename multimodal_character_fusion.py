"""
Moteur de Fusion Multi-Modale pour l'indexation des personnages Disney selon la norme Bolderbast.
Combine :
1. L'OCR textuel du titre (Title Hero imprimé / logo).
2. La détection visuelle de la tête/macaron dans le bandeau supérieur de la page 1.
3. La segmentation des cases et reconnaissance visuelle multi-échelles (DINOv2).
4. L'OCR des dialogues dans les bulles.
5. La cohérence du graphe d'univers Inducks (Ducks vs Mice vs Forest).
6. Les règles strictes de formatage Bolderbast (hero et [xapp:...]).
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from PIL import Image

from page_character_detector import PageCharacterDetector, DEFAULT_INDEX_PATH
from search_character import MAJOR_DISNEY_CHARACTERS

# Univers Inducks principaux
UNIVERSE_DUCKS = {"DD", "US", "HDL", "DA", "FE", "PK", "GL", "GY", "BB", "MDS", "RKD", "LVD", "GD", "GU", "Battista", "Lusky", "FZ"}
UNIVERSE_MICE = {"MM", "GO", "PL", "MI", "CL", "HH", "PE", "PB", "EB", "IP", "CAS", "OH", "TR", "RS"}
UNIVERSE_FOREST = {"BRB", "BW", "JW", "Fifer Pig", "Fiddler Pig", "Practical Pig"}

# Dictionnaire OCR multi-langues des noms de personnages
DIALOGUE_PATTERNS: Dict[str, List[Tuple[re.Pattern, str]]] = {
    "it": [
        (re.compile(r"\b(?:qui\s*,\s*quo\s*(?:,\s*|e\s*)qua|i\s+nipotini)\b", re.I), "HDL"),
        (re.compile(r"\b(?:banda\s+bassotti|i\s+bassotti|bassotti)\b", re.I), "BB"),
        (re.compile(r"\b(?:zio\s*paperone|paperone)\b", re.I), "US"),
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
        (re.compile(r"\b(?:oncle\s*picsou|onc'?\s*picsou|picsou)\b", re.I), "US"),
        (re.compile(r"\b(?:fant[oô]me\s+noir)\b", re.I), "PB"),
        (re.compile(r"\b(?:g[eé]o\s+trouvetou)\b", re.I), "GY"),
        (re.compile(r"\b(?:pat\s+hibulaire)\b", re.I), "PE"),
        (re.compile(r"\b(?:miss\s+tick)\b", re.I), "MDS"),
        (re.compile(r"\b(?:flairsou)\b", re.I), "RKD"),
        (re.compile(r"\b(?:donald\s+duck|donald)\b", re.I), "DD"),
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

_OCR_SINGLETON = None


def get_ocr_engine(lang: str = "it"):
    """Singleton PaddleOCR CPU sécurisé (évite les conflits DLL CUDNN)."""
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        try:
            from paddleocr import PaddleOCR
            _OCR_SINGLETON = PaddleOCR(use_gpu=False, use_angle_cls=False, lang=lang, show_log=False)
        except Exception:
            _OCR_SINGLETON = False
    return _OCR_SINGLETON if _OCR_SINGLETON is not False else None


def extract_ocr_lines(img: Image.Image, lang: str = "it") -> List[str]:
    """Extrait le texte lisible d'une image PIL en CPU."""
    ocr = get_ocr_engine(lang)
    if not ocr:
        return []
    try:
        arr = np.array(img.convert("RGB"))
        res = ocr.ocr(arr, cls=False)
        if res and res[0]:
            return [line[1][0] for line in res[0] if line and line[1]]
    except Exception:
        pass
    return []


class MultiModalCharacterFusion:
    def __init__(self, index_path: str = DEFAULT_INDEX_PATH):
        self.detector = PageCharacterDetector(index_path=index_path)

    def detect_title_hero(
        self,
        first_page_img: Image.Image,
        splash_title_text: str = "",
        lang: str = "it"
    ) -> Tuple[str, str]:
        """
        Détermine le Title Hero (colonne hero Bolderbast) selon la règle stricte :
        1. OCR du texte du titre imprimé dans les 30% supérieurs de la page 1.
        2. Détection visuelle du logo/macaron dans le bandeau supérieur.
        3. Si aucun héros dans le titre : '--'.
        Retourne : (hero_code, titre_detecte)
        """
        pw, ph = first_page_img.size
        title_banner = first_page_img.crop((0, 0, pw, int(ph * 0.32)))

        # 1. OCR textuel direct dans le bandeau de titre
        banner_lines = extract_ocr_lines(title_banner, lang=lang)
        full_banner_text = " ".join(banner_lines) if banner_lines else splash_title_text

        # Priorité aux titres de série isolés / logos imprimés (ex: 'DONALD DUCK', 'ZIO PAPERONE')
        series_title_patterns = [
            (re.compile(r"\b(?:donald\s+duck|donald)\b", re.I), "DD"),
            (re.compile(r"\b(?:zio\s*paperone|uncle\s+scrooge)\b", re.I), "US"),
            (re.compile(r"\b(?:topolino|mickey\s+mouse)\b", re.I), "MM"),
            (re.compile(r"\b(?:paperinik|pikappa|fantomiald)\b", re.I), "PK"),
            (re.compile(r"\b(?:paperoga|fethry)\b", re.I), "FE"),
            (re.compile(r"\b(?:pippo|goofy)\b", re.I), "GO"),
            (re.compile(r"\b(?:paperina|daisy\s+duck)\b", re.I), "DA"),
        ]
        # Vérifier d'abord si une ligne courte de titre contient directement le héros de la série
        for line in banner_lines:
            if len(line.strip()) < 30:
                for pat, code in series_title_patterns:
                    if pat.search(line):
                        return code, line.strip()

        if full_banner_text:
            patterns = DIALOGUE_PATTERNS.get(lang.lower(), DIALOGUE_PATTERNS["it"])
            combined_patterns = patterns + DIALOGUE_PATTERNS.get("fr", []) + DIALOGUE_PATTERNS.get("en", [])
            for pat, code in combined_patterns:
                if pat.search(full_banner_text):
                    return code, full_banner_text

        # 2. Détection visuelle du logo/tête dans le bandeau supérieur
        banner_detections = self.detector.detect_in_panel(title_banner, min_score=0.76)
        if banner_detections:
            return banner_detections[0]["hero"], full_banner_text

        return "--", full_banner_text

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

        # 1. Détermination du Title Hero sur la page 1
        title_hero, title_text = self.detect_title_hero(
            page_images[0], splash_title_text=splash_title_text, lang=lang
        )

        # 2. Détermination de l'univers prioritaire initial (si héros de titre connu)
        dominant_universe = "Ducks"
        if title_hero in UNIVERSE_MICE:
            dominant_universe = "Mice"
        elif title_hero in UNIVERSE_FOREST:
            dominant_universe = "Forest"

        # 3. Analyse visuelle de toutes les cases
        visual_appearances: Dict[str, int] = {}
        visual_panel_appearances: Dict[str, Set[Tuple[int, int]]] = {}
        visual_max_scores: Dict[str, float] = {}
        all_page_detections = []
        all_panels = []

        for p_idx, page_img in enumerate(page_images, start=1):
            page_res = self.detector.analyze_page(page_img, min_score=min_score)
            all_panels.extend(page_res["panels"])

            for det in page_res["all_detections"]:
                h = det["hero"]
                raw_score = det["score"]
                panel_idx = det.get("panel_index", 1)

                # Pondération par l'univers Inducks :
                score = raw_score
                if dominant_universe == "Ducks":
                    if h in UNIVERSE_DUCKS:
                        score += 0.03
                    elif h in UNIVERSE_MICE:
                        score -= 0.15
                elif dominant_universe == "Mice":
                    if h in UNIVERSE_MICE:
                        score += 0.03
                    elif h in UNIVERSE_DUCKS:
                        score -= 0.15

                det["adjusted_score"] = score
                all_page_detections.append(det)

                if score >= min_score:
                    visual_appearances[h] = visual_appearances.get(h, 0) + 1
                    if h not in visual_panel_appearances:
                        visual_panel_appearances[h] = set()
                    visual_panel_appearances[h].add((p_idx, panel_idx))
                    visual_max_scores[h] = max(visual_max_scores.get(h, 0.0), score)

        # 4. OCR des dialogues si non fourni
        ocr_counts: Dict[str, int] = {}
        if dialogue_texts is None:
            dialogue_texts = {}
            for p_idx, page_img in enumerate(page_images, start=1):
                lines = extract_ocr_lines(page_img, lang=lang)
                if lines:
                    dialogue_texts[p_idx] = lines

        if dialogue_texts:
            # Recherche dans TOUTES les langues pour être robuste (fr, it, en, de)
            all_dialogue = " ".join(
                line for lines in dialogue_texts.values() for line in lines
            )
            for lang_key in DIALOGUE_PATTERNS:
                for pat, code in DIALOGUE_PATTERNS[lang_key]:
                    matches = len(pat.findall(all_dialogue))
                    if matches > 0:
                        ocr_counts[code] = ocr_counts.get(code, 0) + matches

        # 5. Consolidation et validation de la liste XAPP
        # Seuil minimal de cases distinctes : 1 case pour splash page unique, 2 cases sinon
        total_panels_count = len(all_panels)
        min_cases = 1 if total_panels_count <= 1 else 2
        validated_xapp: Set[str] = set()

        all_candidates = set(visual_panel_appearances.keys()) | set(ocr_counts.keys())

        for cand in all_candidates:
            v_panels = len(visual_panel_appearances.get(cand, set()))
            v_score = visual_max_scores.get(cand, 0.0)
            t_cnt = ocr_counts.get(cand, 0)

            # 1. Filtre univers strict : Aucune souris dans un univers Ducks sans citation explicite
            if dominant_universe == "Ducks" and cand in UNIVERSE_MICE and t_cnt == 0:
                continue
            if dominant_universe == "Mice" and cand in UNIVERSE_DUCKS and t_cnt == 0:
                continue

            # 2. Personnages non-majeurs (1-shot / longue traîne) : rejetés sauf mention dialogue
            is_major = cand in MAJOR_DISNEY_CHARACTERS
            if not is_major:
                if t_cnt >= 1:
                    validated_xapp.add(cand)
                elif v_panels >= 2 and v_score >= 0.88:
                    validated_xapp.add(cand)
                continue

            # 3. Personnages majeurs de l'univers :
            # Mentionné dans les dialogues (ex: Oncle Picsou ou Daisy cités dans l'histoire)
            if t_cnt >= 1:
                validated_xapp.add(cand)
            # Ou vu visuellement sur au moins min_cases distinctes
            elif v_panels >= min_cases and v_score >= min_score:
                # Éviter la confusion classique PK (Fantomiald) vs DD (Donald)
                if cand == "PK" and title_hero != "PK" and ("DD" in visual_panel_appearances or title_hero == "DD"):
                    continue
                # Éviter la confusion Popop (FE) vs DD si Popop n'est pas cité
                if cand == "FE" and title_hero != "FE" and ("DD" in visual_panel_appearances or title_hero == "DD") and t_cnt == 0:
                    continue
                validated_xapp.add(cand)

        # Règle Dizni #210 : si un Title Hero est défini (différent de '--'), il DOIT être dans xapp
        if title_hero and title_hero != "--":
            validated_xapp.add(title_hero)

        # Tri alphabétique standard Bolderbast
        xapp_sorted = sorted(list(validated_xapp))
        xapp_str = ",".join(xapp_sorted)  # Strictement sans espace : DD,HDL,US

        details = {
            "title_hero": title_hero,
            "title_text": title_text,
            "xapp_str": xapp_str,
            "xapp_tag": f"[xapp:{xapp_str}]" if xapp_str else "",
            "dominant_universe": dominant_universe,
            "visual_appearances": visual_appearances,
            "visual_max_scores": visual_max_scores,
            "ocr_counts": ocr_counts,
            "total_pages": num_pages,
            "panels": all_panels,
            "all_detections": all_page_detections
        }

        return title_hero, xapp_sorted, details
