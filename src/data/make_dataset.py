import pandas as pd
import numpy as np
import os
import argparse
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def make_dataset(input_path, out_dir):
    """
    Exécute le pipeline complet de preprocessing des données.
    """
    logger.info(f"Chargement des données depuis : {input_path}")
    df = pd.read_csv(input_path)
    
    NUMERIC = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite', 'Prix_Revente']
    TARGET = 'Categorie'
    SOURCE_COL = 'Source'

    # 1. Traitement des Outliers (conversion en NaN pour imputation ultérieure)
    logger.info("Étape 1 : Traitement des outliers")
    df_clean = df.copy()
    
    # Poids : -99 indique une valeur inconnue
    df_clean.loc[df_clean['Poids'] == -99, 'Poids'] = np.nan
    
    # Volume : erreurs de signe corrigées avec la valeur absolue
    df_clean.loc[df_clean['Volume'] < 0, 'Volume'] = df_clean['Volume'].abs()
    
    # Opacite : échelle de 0 à 1 (55.0 est une erreur)
    df_clean.loc[df_clean['Opacite'] == 55, 'Opacite'] = np.nan
    
    # Prix_Revente : 9999 ou -50 sont des codes erreurs/manquants
    df_clean.loc[df_clean['Prix_Revente'].isin([9999, -50]), 'Prix_Revente'] = np.nan
    
    logger.info(f"Outliers convertis. NaN totaux : {df_clean[NUMERIC].isnull().sum().sum()}")

    # 2. Imputation finale (IterativeImputer)
    logger.info("Étape 2 : Imputation via IterativeImputer")
    imp_final = IterativeImputer(max_iter=10, random_state=42)
    # On n'impute que les variables numériques
    arr_imputed = imp_final.fit_transform(df_clean[NUMERIC])
    df_imputed = pd.DataFrame(arr_imputed, columns=NUMERIC, index=df_clean.index)

    # 3. Contraintes métier post-imputation
    logger.info("Étape 3 : Application des contraintes métier post-imputation")
    df_imputed['Rigidite'] = df_imputed['Rigidite'].clip(1, 10)
    df_imputed['Opacite']  = df_imputed['Opacite'].clip(0, 1)
    df_imputed['Poids']    = df_imputed['Poids'].clip(lower=0)
    df_imputed['Volume']   = df_imputed['Volume'].clip(lower=0)

    # 4. Standardisation
    logger.info("Étape 4 : Standardisation des variables numériques")
    scaler = StandardScaler()
    df_scaled = df_imputed.copy()
    df_scaled[NUMERIC] = scaler.fit_transform(df_scaled[NUMERIC])

    # 5. Encodage de la colonne Source (One-Hot Encoding)
    logger.info("Étape 5 : Encodage de la variable Source")
    source_series = df_clean[SOURCE_COL].fillna('Inconnu')
    source_dummies = pd.get_dummies(
        source_series,
        prefix='Source',
        drop_first=False,
        dtype=int
    )
    
    # Ajout également d'un Label Encoding pour la source (comme dans le notebook)
    le_source = LabelEncoder()
    source_label_enc = le_source.fit_transform(source_series)

    # Assemblage du DataFrame final
    df_final = df_scaled.copy()
    df_final[source_dummies.columns] = source_dummies.values
    df_final['Source_encoded'] = source_label_enc
    df_final[TARGET] = df_clean[TARGET].values
    
    # Conservation du Rapport_Collecte si présent
    if 'Rapport_Collecte' in df_clean.columns:
        df_final['Rapport_Collecte'] = df_clean['Rapport_Collecte'].values

    # 6. Split Stratifié 70:15:15
    logger.info("Étape 6 : Split des données (Train 70%, Val 15%, Test 15%)")
    df_labeled = df_final[df_final[TARGET].notna()].copy()
    df_unlabeled = df_final[df_final[TARGET].isna()].copy()

    FEATURE_COLS = NUMERIC + source_dummies.columns.tolist() + ['Source_encoded']
    if 'Rapport_Collecte' in df_labeled.columns:
        FEATURE_COLS.append('Rapport_Collecte')
    
    X = df_labeled[FEATURE_COLS]
    y = df_labeled[TARGET]

    # Split Train / Temp (70% - 30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Split Validation / Test (15% - 15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # 7. Sauvegarde des fichiers
    logger.info(f"Étape 7 : Sauvegarde des fichiers dans {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Création de DataFrames complets (X + y) pour faciliter l'usage ultérieur
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    
    df_final.to_csv(os.path.join(out_dir, "dataset_clean.csv"), index=False)
    if not df_unlabeled.empty:
        df_unlabeled.to_csv(os.path.join(out_dir, "dataset_unlabeled.csv"), index=False)

    logger.info("Pipeline terminé avec succès.")
    logger.info(f"Tailles des splits : Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de preprocessing des données Eco-Smart Classifier")
    parser.add_argument("--input", type=str, default="data/raw/dataset_ProjetML_2026.csv", help="Chemin du fichier CSV brut")
    parser.add_argument("--outdir", type=str, default="data/processed", help="Dossier de sortie pour les fichiers nettoyés")
    
    args = parser.parse_args()
    
    make_dataset(args.input, args.outdir)
