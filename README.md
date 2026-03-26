# Projet MVA DLMI - Histopathology OOD Classification

Ce README explique rapidement a quoi sert chaque notebook du depot pour que ce soit simple a reprendre.

## Donnees

- Les donnees en format `.h5` ne sont pas presentes sur GitHub.
- Il faut donc recuperer ces fichiers de donnees separement pour pouvoir executer le pipeline complet.

## Notebooks du projet

### `getting_started.ipynb`

- Notebook fourni par les profs.
- Je ne l'ai pas modifie.
- Je l'ai seulement execute une fois pour verifier que tout fonctionne.

### `MVA_DLMI_adapters.ipynb`

- TD corrige (support de reference).
- Sert surtout a s'inspirer de la methode, de la structure et des idees.

### `TP-validation_teacher-version.ipynb`

- Autre notebook corrige / version enseignant.
- Utile comme base de comparaison ou pour verifier des choix techniques.

### `explore_data.ipynb`

- Notebook que j'ai fait pour me familiariser avec les donnees.
- Objectif: explorer le dataset, observer des exemples et verifier des points de preprocessing.
- Il sert a comprendre "ce qu'il se passe" avant de coder le pipeline principal.
- Contenu detaille:
  - configuration/imports et verification que les fichiers attendus sont presents;
  - inspection de la structure HDF5 (comment sont stockes les patches et labels);
  - creation d'une table recap par patch pour faciliter l'analyse;
  - synthese quantitative globale (comptages, repartitions);
  - graphiques d'equilibre des classes et de repartition par centre;
  - analyse des pixels sur un sous-echantillon:
    - histogrammes d'intensite par canal,
    - correlations entre canaux,
    - effet d'une normalisation z-score par patch,
    - recherche des patches les plus clairs / plus sombres;
  - comparaison du profil moyen des canaux entre validation et centres du train;
  - visualisations qualitatives:
    - une image par couple (centre, label) pour train/val,
    - grille aleatoire sur le domaine test.
- En pratique, ce notebook sert a:
  - verifier qu'il n'y a pas d'incoherence evidente dans les donnees;
  - guider les choix de preprocessing/augmentation avant l'entrainement;
  - documenter les intuitions sur le shift de domaine et l'equilibre des classes.

### `my_pipeline.ipynb`

- Notebook principal du projet.
- C'est ici qu'on va travailler, tester et faire evoluer la solution.
- C'est un "experiment runner" centre sur DINOv2 + adapters.
- Organisation du notebook:
  - imports, chemins et utilitaires;
  - bloc config (section a modifier en priorite pour lancer des experiences);
  - dataset + preprocessing;
  - construction du modele (backbone gele + adapter optionnel + tete de classification);
  - boucle d'entrainement avec logs et checkpoints;
  - prediction optionnelle sur `test.h5`;
  - runner d'experiences (definition de plusieurs runs puis lancement).
- Role concret:
  - centraliser les essais dans un seul notebook;
  - comparer facilement plusieurs configurations;
  - produire des checkpoints et des sorties exploitables pour l'evaluation.
- Recommandation de travail:
  - utiliser `explore_data.ipynb` pour valider les hypotheses data;
  - implementer/ajuster dans `my_pipeline.ipynb`;
  - garder les notebooks profs/corriges comme references en cas de doute.

## Remarque

- Les fichiers `.csv` et `.pth` ne sont pas decrits ici (intentionnellement).
