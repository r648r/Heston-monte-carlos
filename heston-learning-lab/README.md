# 🎓 Heston Learning Lab - Laboratoire Interactif d'Apprentissage

Bienvenue dans le **Heston Learning Lab**, un environnement pédagogique interactif pour comprendre en profondeur le modèle de Heston et la finance quantitative!

## 📚 À Propos

Ce laboratoire vous guide pas à pas à travers les concepts mathématiques et l'implémentation pratique du **modèle de Heston** pour la modélisation des prix d'actifs avec volatilité stochastique.

### 🎯 Pour qui ?

- Étudiants en finance quantitative
- Traders/investisseurs cherchant à comprendre les modèles stochastiques
- Développeurs en fintech
- Toute personne curieuse des mathématiques financières **sans prérequis avancés** !

### ✨ Caractéristiques

- **5 notebooks Jupyter progressifs** expliquant tous les concepts depuis zéro
- **Explications intuitives** avec analogies du monde réel
- **Visualisations interactives** pour chaque concept
- **Exemples pratiques** avec Bitcoin/crypto
- **Code Python commenté** ligne par ligne
- **Génération de rapports HTML** professionnels comme dans Heston.v2

## 📖 Contenu des Notebooks

### 01 - Introduction aux Concepts de Base
- Variables aléatoires et distributions
- Simulation simple de prix
- Comprendre la volatilité
- Premiers graphiques interactifs

**Durée:** ~30 minutes | **Niveau:** Débutant

### 02 - Mouvement Brownien
- Le mouvement Brownien standard
- Propriétés mathématiques (avec vérifications empiriques)
- Mouvements Browniens corrélés
- Lien avec les prix d'actifs

**Durée:** ~45 minutes | **Niveau:** Intermédiaire

### 03 - Modèle de Heston Complet
- Formulation mathématique
- Processus CIR pour la variance
- Implémentation complète en Python
- Impact des paramètres

**Durée:** ~60 minutes | **Niveau:** Intermédiaire

### 04 - Simulations Monte Carlo
- Génération de milliers de trajectoires
- Calcul de statistiques (percentiles, probabilités)
- Analyse de sensibilité
- Visualisations avancées

**Durée:** ~45 minutes | **Niveau:** Intermédiaire

### 05 - Génération de Rapports HTML
- Création de graphiques publication-ready
- Rapport HTML interactif complet
- Workflow de bout en bout
- Style professionnel comme Heston.v2

**Durée:** ~40 minutes | **Niveau:** Avancé

## 🚀 Installation et Démarrage

### Option 1: Avec environnement virtuel (Recommandé)

```bash
# 1. Cloner ou naviguer vers le répertoire
cd heston-learning-lab

# 2. L'environnement virtuel est déjà créé (.heston-venv)
# Si ce n'est pas le cas:
python3 -m venv .heston-venv

# 3. Activer l'environnement
source .heston-venv/bin/activate  # Mac/Linux
# OU
.heston-venv\\Scripts\\activate  # Windows

# 4. Les packages sont déjà installés
# Si besoin de réinstaller:
pip install -r requirements.txt

# 5. Lancer Jupyter Lab
jupyter lab
```

### Option 2: Avec Docker

```bash
# 1. Construire l'image Docker
docker build -t heston-lab .

# 2. Lancer le container
docker run -p 8888:8888 -v $(pwd)/notebooks:/workspace/notebooks heston-lab

# 3. Ouvrir le lien affiché dans le terminal
# Exemple: http://localhost:8888
```

### Option 3: Avec Docker Compose

```bash
# Lancer avec docker-compose
docker-compose up

# Arrêter
docker-compose down
```

## 🎮 Utilisation

1. **Ouvrez Jupyter Lab** (http://localhost:8888)
2. **Commencez par le notebook 01** et suivez l'ordre
3. **Exécutez les cellules** une par une pour voir les résultats
4. **Expérimentez** ! Changez les paramètres, observez les effets
5. **Lisez les explications** entre les cellules de code

## 📊 Exemple de Résultat

À la fin du notebook 05, vous aurez généré un **rapport HTML professionnel** contenant:

- Distribution des prix simulés
- Trajectoires Monte Carlo
- Statistiques complètes (percentiles, probabilités)
- Paramètres du modèle expliqués
- Visualisations interactives

![Exemple de rapport](https://via.placeholder.com/800x400.png?text=Rapport+Heston+HTML)

## 🔧 Technologies Utilisées

- **Python 3.11+**
- **Jupyter Lab** - Interface interactive
- **NumPy** - Calculs numériques
- **SciPy** - Statistiques
- **Matplotlib/Seaborn** - Visualisations
- **Pandas** - Manipulation de données
- **Sympy** - Mathématiques symboliques

## 📁 Structure du Projet

```
heston-learning-lab/
├── notebooks/
│   ├── 01_Introduction_Concepts_Base.ipynb
│   ├── 02_Mouvement_Brownien.ipynb
│   ├── 03_Modele_Heston_Complet.ipynb
│   ├── 04_Simulations_Monte_Carlo.ipynb
│   └── 05_Generation_Rapports_HTML.ipynb
├── .heston-venv/          # Environnement virtuel (caché)
├── Dockerfile              # Pour Docker
├── docker-compose.yml      # Pour Docker Compose
├── requirements.txt        # Dépendances Python
└── README.md              # Ce fichier
```

## 🎓 Concepts Mathématiques Couverts

### Niveau Fondamental
- Variables aléatoires
- Distribution normale (Gaussienne)
- Espérance, variance, écart-type
- Percentiles et quantiles

### Niveau Intermédiaire
- Processus stochastiques
- Mouvement Brownien
- Équations différentielles stochastiques (EDS)
- Corrélation entre processus

### Niveau Avancé
- Modèle de Heston
- Processus CIR (Cox-Ingersoll-Ross)
- Volatilité stochastique
- Méthode de Monte Carlo
- Discrétisation d'Euler-Maruyama

## 💡 Conseils d'Utilisation

### Pour les Débutants

1. **Ne sautez pas d'étapes** - Chaque notebook construit sur le précédent
2. **Prenez votre temps** - Compréhension > Vitesse
3. **Expérimentez** - Changez les paramètres pour voir les effets
4. **Posez-vous des questions** - "Et si je change ce paramètre ?"

### Pour les Plus Avancés

1. **Comparez avec Black-Scholes** - Notebook 03 contient une comparaison
2. **Testez différents scénarios** - Bull market, bear market, high volatility
3. **Calibrez sur données réelles** - Utilisez des données de votre choix
4. **Étendez le code** - Ajoutez des fonctionnalités (jumps, régimes, etc.)

## 🎯 Exercices Suggérés

### Faciles
1. Changer les paramètres du modèle et observer l'impact
2. Simuler un autre actif (actions, ETH, etc.)
3. Modifier les horizons de temps (7j, 60j, 1 an)

### Moyens
1. Implémenter un test de la condition de Feller
2. Calculer le smile de volatilité implicite
3. Comparer Heston avec un modèle GARCH

### Difficiles
1. Calibrer les paramètres sur données historiques réelles
2. Implémenter le pricing d'options européennes
3. Ajouter des sauts (modèle de Bates)

## 🔗 Liens avec Heston.v2

Ce laboratoire est conçu pour compléter votre projet **Heston.v2** :

- **Heston.v2** : Production-ready, optimisé, pour le trading réel
- **Heston Learning Lab** : Pédagogique, explications détaillées, apprentissage

Vous pouvez utiliser ce lab pour:
- Comprendre le code de Heston.v2
- Tester de nouveaux paramètres avant de les utiliser en production
- Former des collaborateurs
- Documenter votre stratégie

## 📚 Ressources Additionnelles

### Articles Académiques
- **Heston (1993)** - "A Closed-Form Solution for Options with Stochastic Volatility"
- **Cox-Ingersoll-Ross (1985)** - "A Theory of the Term Structure of Interest Rates"

### Livres
- "The Volatility Surface" - Jim Gatheral
- "Stochastic Volatility Modeling" - Lorenzo Bergomi
- "Python for Finance" - Yves Hilpisch

### Cours en Ligne
- Coursera: "Financial Engineering and Risk Management"
- QuantLib documentation
- Financial-Models-Numerical-Methods (repo inspirant ce projet)

## 🤝 Contribution

Ce projet est à but pédagogique. N'hésitez pas à:
- Améliorer les explications
- Ajouter des exemples
- Corriger des erreurs
- Proposer de nouveaux notebooks

## ⚠️ Avertissement

Ce laboratoire est à but **éducatif uniquement**. Les modèles présentés sont des simplifications de la réalité.

**NE PAS UTILISER** directement pour:
- Trading réel sans validation approfondie
- Gestion de fonds sans tests rigoureux
- Conseils financiers

Toujours:
- Backtester sur données historiques
- Valider avec des experts
- Comprendre les limites du modèle

## 📝 Licence

Ce projet est fourni "tel quel" à des fins éducatives.

## 🙏 Remerciements

Inspiré par:
- **Financial-Models-Numerical-Methods** - Excellent repo de référence
- **Votre projet Heston.v2** - Implémentation production
- La communauté Python finance

---

## 🚀 Commencer Maintenant!

```bash
# Activer l'environnement
source .heston-venv/bin/activate

# Lancer Jupyter
jupyter lab

# Ouvrir 01_Introduction_Concepts_Base.ipynb et c'est parti ! 🎉
```

**Bon apprentissage ! 📊🎓**

---

*Dernière mise à jour: 2024-11-24*
