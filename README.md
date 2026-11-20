# 🎨 Disney Vision - Détection & Identification des Personnages Disney

Module d'intelligence visuelle pour la détection et la reconnaissance automatique des personnages de bandes dessinées Disney (comics), conçu pour compléter **dbi-generator** lorsqu'un code officiel Inducks n'est pas identifié.

---

## 🚀 Architecture Technique

Le système repose sur un pipeline en **3 étages** :

1. **Segmentation adaptative des cases (Panel Extraction)** :
   * Découpage géométrique des bandes et des cases par analyse des gouttières de marge en tenant compte de la colorimétrie réelle du papier de scan (`panel_extractor.py`).
2. **Localisation des personnages** :
   * Détection des cibles vivantes dans chaque case à l'aide de **YOLOv8** (`ultralytics`).
3. **Identification par Plongement Métrique (DINOv2 + K-NN Search)** :
   * Extraction de l'empreinte vectorielle (vecteur normalisé de dimension 384) avec le modèle de fondation auto-supervisé **DINOv2** de Meta (`dinov2_vits14`).
   * Comparaison instantanée (< 1 ms) par produit scalaire / similarité cosinus avec un index pré-calculé de 2 000 portraits de référence couvrant les 250 personnages les plus récurrents d'Inducks.

---

## 📂 Organisation des Fichiers

| Fichier | Rôle |
| :--- | :--- |
| **`download_character_references.py`** | Télécharge les vignettes de portraits officielles depuis le catalogue Inducks (`characterscans.dbp`). |
| **`build_character_embeddings.py`** | Calcule les embeddings DINOv2 de toutes les images de référence et génère l'index compact `.npz`. |
| **`search_character.py`** | Identifie un personnage à partir d'un crop de visage ou buste avec les scores de similarité. |
| **`page_character_detector.py`** | Analyse une planche de BD complète (découpe les cases, localise les personnages, agrège les votes). |
| **`bridge.py`** | Passerelle d'intégration directe avec `dbi-generator` pour renseigner automatiquement le `hero` et `xapp`. |
| **`data/character_embeddings.npz`** | Matrice des vecteurs d'embeddings (1 999 $\times$ 384 dimensions, 2.74 Mo). |
| **`data/references/`** | Dossier local des images de référence classées par code de personnage Inducks. |

---

## 🛠️ Utilisation Rapide (CLI)

### 1. Identifier un personnage sur un crop
```bash
python search_character.py "chemin/vers/crop.png"
```

### 2. Analyser une planche de scan complète
```bash
python page_character_detector.py "chemin/vers/scan_page.png"
```

### 3. Étendre l'index à d'autres personnages
```bash
# Télécharger jusqu'à 8 scans pour les 500 premiers personnages :
python download_character_references.py --top-chars 500 --max-per-char 8

# Reconstruire l'index DINOv2 :
python build_character_embeddings.py
```
