# 🧠 TP2 : Modèles Génératifs avec PyTorch
## Modèle Discriminatif, Auto-encodeur Simple (AE) et Variational Autoencoder (VAE)

Ce répertoire présente la réalisation complète du **TP2 sur les Modèles Génératifs avec PyTorch**, mené sur le jeu de données d'images manuscrites **MNIST**.  
Le travail suit une progression pédagogique en 4 étapes majeures allant du modèle discriminatif supervisé classique jusqu'aux modèles génératifs variationnels probabilistes.

---

## 👨‍💻 Auteur & Cadre
- **Étudiant :** Iliass Mouhtad
- **Cadre académique :** Module d'Intelligence Artificielle Multimodale / Deep Learning
- **Établissement :** EMSI
- **Notebook de référence :** `Modeles_generatifs_pytorch_avec_explications.ipynb` (Pr. Mohamed YOUSSFI)

---

## 🎯 Objectifs Réalisés dans le Notebook

1. **Chargement et Inspection du Dataset MNIST :**
   - Téléchargement et normalisation des tenseurs (`transforms.ToTensor()` dans $[0.0, 1.0]$).
   - Organisation par mini-lots (`DataLoader` avec `batch_size = 128`).
   - Visualisation d'un lot d'images $28 \times 28$ avec étiquettes réelles.

2. **Modèle Discriminatif (Classification Supervisée) :**
   - Estimation de la distribution conditionnelle $P(y|x)$.
   - Architecture MLP : couche d'aplatissement `Flatten` $\rightarrow$ linéaire dense $784 \rightarrow 128$ avec `ReLU` $\rightarrow$ sortie logits $128 \rightarrow 10$.
   - Optimisation avec `nn.CrossEntropyLoss` et `torch.optim.Adam` ($\text{lr} = 10^{-3}$) sur 5 époques.
   - Évaluation de la précision : **Accuracy de 97.32%** sur les 10 000 images de test (validée avec PyTorch natif et `torcheval.metrics.MulticlassAccuracy`).
   - Analyse critique : incapacité d'un modèle discriminatif à générer des données ($P(x)$ non modélisée).

3. **Auto-encodeur Simple (Apprentissage Non Supervisé & Compression) :**
   - Principe : compression $x \rightarrow z$ puis décompression $z \rightarrow \hat{x} \approx x$.
   - Goulot d'étranglement (*bottleneck*) compact en **2 dimensions** (`latent_dim = 2`) pour une visualisation directe sans réduction dimensionnelle additionnelle.
   - Fonction de coût MSE (`nn.MSELoss`), convergence à une perte de reconstruction de **0.0445**.
   - Évaluation visuelle côte à côte : images originales vs images reconstruites.
   - Exploration de l'espace latent 2D : projection du test set en nuage de points $(z_1, z_2)$ coloré par chiffre (clusters distincts).
   - Synthèse ciblée par calcul du **centroïde moyen** de chaque classe.
   - Démonstration des limites de l'AE : espace discontinu avec "trous", échec de la génération aléatoire $z \sim \mathcal{N}(0, I)$ produisant des motifs flous/incohérents.

4. **Variational Autoencoder - VAE (Modèle Génératif Probabiliste) :**
   - Formalisme bayésien : encodage vers une loi a posteriori $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$ et a priori $p(z) \sim \mathcal{N}(0, I)$.
   - Mécanisme du **Reparameterization Trick** :
     $$z = \mu + \sigma \odot \epsilon \quad \text{avec} \quad \epsilon \sim \mathcal{N}(0, I), \quad \sigma = \exp\left(\frac{1}{2}\log\sigma^2\right)$$
   - Fonction de perte **ELBO** (Evidence Lower Bound) :
     $$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{BCE}}(x, \hat{x}) + \beta \cdot D_{\text{KL}}\left(q_\phi(z|x) \,\|\, \mathcal{N}(0, I)\right)$$
   - Entraînement sur 15 époques avec suivi décomposé (`Perte totale = 145.31`, `Reconstruction BCE = 139.01`, `Divergence KL = 6.30`).
   - Visualisation de l'espace latent régularisé $(\mu_1, \mu_2)$ : centré en $(0, 0)$, dispersion unitaire, sans vides béants.
   - **Génération ex nihilo :** génération de 20 nouveaux chiffres inédits à partir de bruit gaussien pur $z \sim \mathcal{N}(0, I)$.
   - **Exploration continue de la variété latente 2D :** balayage d'une grille régulière $15 \times 15$ sur $[-3, 3] \times [-3, 3]$ montrant des transitions morphologiques fluides entre chiffres.
   - **Interpolation latente (Morphing) :** transition linéaire $z(t) = (1 - t)z_A + tz_B$ continue entre deux chiffres réels.

---

## 📊 Tableau Synthétique Comparatif des 3 Paradigmes

| Propriété / Critère | 1. Modèle Discriminatif (MLP) | 2. Auto-encodeur Simple (AE) | 3. Variational Autoencoder (VAE) |
| :--- | :--- | :--- | :--- |
| **Objectif principal** | Classifier : estimer $P(y\|x)$ | Compresser et reconstruire : $\hat{x} \approx x$ | Modéliser la distribution des données : $P(x)$ |
| **Type d'apprentissage** | Supervisé (requiert les étiquettes $y$) | Non supervisé (sans étiquettes) | Non supervisé probabiliste (sans étiquettes) |
| **Espace latent** | Aucun espace continu | Déterministe ($z = f(x)$), discontinu avec zones vides | Probabiliste ($z \sim \mathcal{N}(\mu, \sigma^2)$), continu et centré en $0$ |
| **Fonction de perte** | Cross-Entropy Loss | Mean Squared Error (MSE) | ELBO = Reconstruction (BCE) + Divergence KL |
| **Résultats obtenus** | **Accuracy : 97.32%** | **MSE Loss : 0.0445** | **ELBO Loss : 145.31** (Rec: 139.01, KL: 6.30) |
| **Génération ex nihilo ($z \sim \mathcal{N}(0, I)$)** | ❌ Impossible | ❌ Formes floues et discontinues | ✅ Chiffres nets, variés et réalistes |
| **Continuité de l'espace** | Non applicable | Discontinu (zones vides non entraînées) | Variété continue (grille $15 \times 15$ & morphing fluide) |

---

## 📁 Fichiers du TP2

| Fichier | Description |
|---|---|
| **[`TP2_Modeles_Generatifs.ipynb`](TP2_Modeles_Generatifs.ipynb)** | **Notebook Jupyter principal** contenant l'intégralité du code exécuté (45 cellules), les explications mathématiques détaillées et les 12 figures intégrées. |
| **[`README.md`](README.md)** | Documentation synthétique du TP2, présentation des architectures et rapport des résultats. |

---

## ⚙️ Instructions d'Exécution

### 1. Prérequis
- Python 3.10 ou supérieur
- PyTorch et torchvision
- Jupyter Notebook / JupyterLab ou VS Code

```bash
pip install torch torchvision matplotlib numpy torcheval
```

### 2. Lancement du Notebook
```bash
# Se placer dans le dossier tp2
cd tp2

# Lancer le notebook
jupyter notebook TP2_Modeles_Generatifs.ipynb
```
