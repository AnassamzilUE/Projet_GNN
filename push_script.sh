
set -e

echo "--- ÉTAPE 1: DESTRUCTION DU DÉPÔT CORROMPU ---"
rm -rf .git
echo "Ancien dépôt .git supprimé."

echo "--- ÉTAPE 2: CRÉATION D'UN NOUVEAU DÉPÔT SAIN ---"
git init
git lfs install
echo "Nouveau dépôt initialisé et LFS installé."

echo "--- ÉTAPE 3: CORRECTION IMMÉDIATE DES PERMISSIONS ---"
chmod +x .git/hooks/*
echo "Permissions d'exécution ajoutées. Vérification :"
ls -l .git/hooks

echo "--- ÉTAPE 4: CONFIGURATION DE LFS ET .gitignore ---"
git lfs track "*.pt"
git lfs track "*.csv"
echo "CARE-GNN/" >> .gitignore
echo "Suivi LFS configuré."

echo "--- ÉTAPE 5: LE PREMIER COMMIT PROPRE---"
git add .
git commit -m "fix(Ultime): Initialisation propre et atomique du projet"
echo "Commit créé avec succès."

echo "--- ÉTAPE 6: CONNEXION ET ENVOI À GITHUB ---"
git branch -M main
git remote add origin https://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@github.com/AnassamZILUE/Projet_GNN.git
git push -u origin main
echo "--- SCRIPT TERMINÉ AVEC SUCCÈS ! ---"
