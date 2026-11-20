"""
Évaluation de la fiabilité du système de reconnaissance par échantillonnage
sur les annotations manuelles des contributeurs Inducks.
Compare les prédictions (Top-1, Top-3, et xapp) avec le ground truth humain.
"""

import os
import random
import json
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Tuple

from search_character import CharacterMatcher, DEFAULT_INDEX_PATH


def run_evaluation(num_samples: int = 100, index_path: str = DEFAULT_INDEX_PATH):
    print(f"Chargement de l'index {index_path}...")
    matcher = CharacterMatcher(index_path=index_path)

    # Récupérer les métadonnées de l'index
    metadata = matcher.metadata
    embeddings = matcher.embeddings
    total_refs = len(metadata)

    # Grouper par personnage
    by_hero: Dict[str, List[int]] = {}
    for idx, meta in enumerate(metadata):
        by_hero.setdefault(meta["hero"], []).append(idx)

    # Ne garder que les personnages ayant au moins 2 images pour permettre le test Leave-One-Out
    eval_candidates = [h for h, idxs in by_hero.items() if len(idxs) >= 2]
    print(f"Personnages testables (>= 2 images) : {len(eval_candidates)} sur {len(by_hero)}")

    # Échantillonnage aléatoire reproductible
    random.seed(42)
    selected_heroes = random.sample(eval_candidates, min(num_samples, len(eval_candidates)))

    top1_correct = 0
    top3_correct = 0
    total_tested = 0

    results_log = []

    print(f"\nLancement du benchmark sur {len(selected_heroes)} images testées...")

    for rank_idx, hero in enumerate(selected_heroes, start=1):
        idxs = by_hero[hero]
        test_idx = random.choice(idxs)
        test_meta = metadata[test_idx]
        test_vec = embeddings[test_idx]

        # Leave-One-Out : on calcule la similarité cosinus avec tout l'index SAUF l'image de test elle-même
        similarities = np.dot(embeddings, test_vec)
        similarities[test_idx] = -1.0  # Exclure l'image elle-même

        # Agrégation par héros (score max)
        hero_scores: Dict[str, float] = {}
        for idx, score in enumerate(similarities):
            h = metadata[idx]["hero"]
            if h not in hero_scores or score > hero_scores[h]:
                hero_scores[h] = float(score)

        sorted_preds = sorted(hero_scores.items(), key=lambda x: x[1], reverse=True)
        top1_hero, top1_score = sorted_preds[0]
        top3_heroes = [h for h, _ in sorted_preds[:3]]

        is_top1 = (top1_hero == hero)
        is_top3 = (hero in top3_heroes)

        if is_top1:
            top1_correct += 1
        if is_top3:
            top3_correct += 1
        total_tested += 1

        status = "✅ EXACT" if is_top1 else ("⚠️ TOP-3" if is_top3 else "❌ ERREUR")
        results_log.append({
            "expected": hero,
            "top1": top1_hero,
            "top1_score": round(top1_score, 4),
            "top3": top3_heroes,
            "status": status,
            "file": test_meta["filename"]
        })

        if rank_idx % 20 == 0 or rank_idx == len(selected_heroes):
            print(f"  Progression: {rank_idx:3d}/{len(selected_heroes)} | Top-1: {top1_correct/total_tested*100:.1f}% | Top-3: {top3_correct/total_tested*100:.1f}%")

    top1_acc = (top1_correct / total_tested) * 100
    top3_acc = (top3_correct / total_tested) * 100

    print("\n" + "=" * 60)
    print("           RÉSULTATS DE L'ÉVALUATION (GROUND TRUTH INDUCKS)           ")
    print("=" * 60)
    print(f"Images de test évaluées (Leave-One-Out) : {total_tested}")
    print(f"Précision Top-1 (Héros exact trouvé)    : {top1_acc:.2f}% ({top1_correct}/{total_tested})")
    print(f"Précision Top-3 (Dans le top 3)         : {top3_acc:.2f}% ({top3_correct}/{total_tested})")
    print("=" * 60)

    # Afficher quelques exemples représentatifs
    print("\nExemples de correspondances :")
    for r in results_log[:10]:
        print(f"  {r['status']:<10} Attendu: {r['expected']:<18} | Prédit: {r['top1']:<18} (score: {r['top1_score']*100:.1f}%)")

    return top1_acc, top3_acc


if __name__ == "__main__":
    run_evaluation(num_samples=100)
