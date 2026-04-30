# 📝 Journal des Interactions IA — Eco-Smart Classifier

> Ce fichier documente toutes les interactions avec les outils IA (Antigravity/Claude) 
> utilisés durant le développement du projet, conformément à la Charte IA du Cahier des Charges.

## Légende des Zones
- 🔴 Rouge : IA interdite — code écrit par l'étudiant
- 🟠 Orange : IA structuration seulement
- 🟢 Vert : IA libre

| Date | Zone | Module | Prompt utilisé | Réponse IA résumée | Décision finale | Esprit critique |
|------|------|--------|---------------|-------------------|-----------------|-----------------|
| 2026-03-22 | 🟠 Orange | Init | "Aide à structurer le projet selon le cahier des charges" | Proposition arborescence complète DVC/src/tests | Architecture validée et adoptée | Structure conforme au CDC — aucune modification nécessaire |
| 2026-03-22 | 🟢 Vert | MLOps | "Mettre en place FastAPI et Docker" | Squelette FastAPI + Dockerfile + docker-compose | Adopté avec modifications des chemins | L'IA a proposé des chemins absolus — corrigé en chemins relatifs |
| 2026-03-23 | 🟠 Orange | Module 1 | "Débugger le notebook EDA" | Conseils sur structure notebooks et bonnes pratiques | Corrections appliquées sélectivement | Certaines suggestions trop génériques — adaptées au dataset spécifique |
| 2026-03-23 | 🟢 Vert | Module 1 | "Implémenter KNN Imputer et IterativeImputer avec comparaison RMSE" | Code comparaison 3 imputeurs | Adopté — méthode RMSE plus rigoureuse que variance | L'IA a proposé variance — nous avons choisi RMSE car plus interprétable |
| 2026-03-24 | 🟢 Vert | Module 2 | "Optimisation hyperparamètres avec Optuna" | Pipeline Optuna + 5 trials | Adopté avec n_trials augmenté à 10 | L'IA proposait 5 trials — insuffisant pour une vraie optimisation |
| 2026-03-24 | 🟢 Vert | Module 2 | "Implémenter SHAP values pour explicabilité" | TreeExplainer + summary_plot | Adopté — visualisation claire | Résultats SHAP cohérents avec intuition métier |
| 2026-03-24 | 🟢 Vert | Module 2 | "Stacking de modèles RF + XGBoost + LR" | StackingClassifier sklearn | Adopté — amélioration accuracy | L'IA a bien suggéré LogisticRegression comme méta-modèle |
| 2026-03-25 | 🟢 Vert | Module 3 | "Clustering KMeans + méthode du coude + PCA 2D" | Code complet avec silhouette score | Adopté avec ajout interprétation physique des clusters | L'IA n'avait pas prévu l'interprétation métier — ajoutée manuellement |
| 2026-03-25 | 🟢 Vert | Module 5 | "Pipeline Multimodal ColumnTransformer" | StandardScaler + TF-IDF + RandomForest | Adopté — architecture claire | French stopwords ajoutés manuellement — l'IA utilisait English |
| 2026-03-26 | 🟢 Vert | Module 6 | "MLflow tracking 7 expériences" | 7 runs avec métriques + artifacts | Adopté avec ajout confusion matrix | L'IA oubliait les confusion matrices comme artifacts — ajoutées |
| 2026-03-26 | 🟢 Vert | MLOps | "Dockerfile multi-stage" | Builder + Runtime python:3.10-slim | Adopté — image légère | L'IA n'avait pas inclus data/processed — ajouté manuellement |
| 2026-03-26 | 🟢 Vert | MLOps | "GitHub Actions CI/CD" | Lint + pytest + coverage + Docker build | Adopté avec --cov-fail-under=70 | Seuil coverage ajouté — l'IA ne l'avait pas prévu |

## 🔴 Code écrit sans IA (Zone Rouge)

Conformément à la Charte IA, ces éléments ont été implémentés sans assistance IA :
- Analyse MCAR/MAR/MNAR et Little's MCAR test
- Justifications métier des outliers (Poids=-99, Opacite=55, etc.)
- Interprétation physique des clusters
- Choix et justification des stratégies d'imputation

## 💡 Bilan Esprit Critique

Les principales corrections apportées aux suggestions IA :
1. **Stopwords** : L'IA utilisait English → corrigé en French (NLTK)
2. **RMSE vs Variance** : L'IA proposait variance → RMSE plus rigoureux
3. **n_trials Optuna** : L'IA proposait 5 → augmenté à 10
4. **Chemins relatifs** : L'IA utilisait chemins absolus → corrigé
5. **Interprétation clusters** : L'IA ne l'avait pas prévu → ajoutée
