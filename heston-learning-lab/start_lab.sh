#!/bin/bash

echo "🎓 Heston Learning Lab - Démarrage"
echo "=================================="
echo ""

# Vérifier si l'environnement virtuel existe
if [ ! -d ".heston-venv" ]; then
    echo "⚠️  Environnement virtuel non trouvé. Création..."
    python3 -m venv .heston-venv
    echo "✅ Environnement créé"
fi

# Activer l'environnement
echo "🔄 Activation de l'environnement virtuel..."
source .heston-venv/bin/activate

# Vérifier si les packages sont installés
if ! python -c "import jupyter" 2>/dev/null; then
    echo "📦 Installation des dépendances..."
    pip install -r requirements.txt
    echo "✅ Dépendances installées"
else
    echo "✅ Dépendances déjà installées"
fi

echo ""
echo "🚀 Lancement de Jupyter Lab..."
echo "   Ouvrez votre navigateur à: http://localhost:8888"
echo ""
echo "   Pour arrêter: Ctrl+C"
echo ""

# Lancer Jupyter Lab
jupyter lab --no-browser
