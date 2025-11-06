import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# --- 1. Configuration et Chargement des Données ---
OUTPUT_FILE = 'transformed_dataset.csv' # Le fichier de features créé précédemment
df = pd.read_csv(OUTPUT_FILE)

# Caractéristiques utilisées (celles qui étaient dans le Top 10 d'importance)
FEATURE_COLUMNS = [
    'Motor_Roll_Diff_Mean_DS', 'AccZ_IMU1_RMS_DS', 'AccZ_IMU1_Skewness_DS', 
    'Roll_Control_Error_Var', 'IMU_Roll_Diff_Std', 'IMU_Pitch_Diff_Std',
    'FFT_AccZ_Peak1_Amp', 'FFT_AccZ_Peak1_Freq', 'FFT_AccZ_Peak2_Amp', 
    'FFT_Energy_MotorBand', 'FFT_Energy_HarmonicBand', 'RefThrust_Mean_DS'
]

# --- 2. Préparation des Données pour la Détection d'Anomalies ---

# Séparation des données Saines vs. Défectueuses
df_sain = df[df['TARGET_CLASS'] == 'F0SV0'].copy()
df_defaut = df[df['TARGET_CLASS'] != 'F0SV0'].copy()

# Ensemble d'entraînement: UNIQUEMENT les données saines
X_train_sain = df_sain[FEATURE_COLUMNS]

# Ensemble de test: Toutes les données pour l'évaluation
X_test = df[FEATURE_COLUMNS]
# Création de l'étiquette binaire de VÉRITÉ TERRAIN pour l'évaluation: 0 = Sain (Inlier), 1 = Défaut (Outlier)
y_true_binary = (df['TARGET_CLASS'] != 'F0SV0').astype(int) 

print(f"Échantillons d'entraînement (Sain): {X_train_sain.shape[0]}")
print(f"Échantillons de test (Total): {X_test.shape[0]}")

# --- 3. Standardisation des Features ---
# La mise à l'échelle est cruciale pour les modèles basés sur la distance
scaler = StandardScaler()
# IMPORTANT: On ajuste le scaler UNIQUEMENT sur les données saines d'entraînement
X_train_sain_scaled = scaler.fit_transform(X_train_sain)
X_test_scaled = scaler.transform(X_test) # On applique la transformation aux données de test

# --- 4. Entraînement de l'Isolation Forest ---

# L'Isolation Forest s'entraîne sur le concept que les anomalies sont faciles à "isoler".
# Le paramètre 'contamination' est la proportion attendue d'anomalies dans le jeu d'entraînement.
# Puisque nous entraînons UNIQUEMENT sur des données saines, nous fixons contamination à une petite valeur (ex: 1%)
iso_forest = IsolationForest(
    n_estimators=100, 
    contamination=0.01, 
    random_state=42, 
    max_samples='auto'
)

print("\nDébut de l'entraînement de l'Isolation Forest (sur les données Saines F0SV0)...")
iso_forest.fit(X_train_sain_scaled)
print("Entraînement terminé.")

# --- 5. Évaluation des Scores d'Anomalie ---

# Le Isolation Forest retourne un score de décision:
# Score positif ou proche de zéro: Inlier (Normal/Sain)
# Score négatif: Outlier (Anomalie/Défaut)

df['anomaly_score'] = iso_forest.decision_function(X_test_scaled)

# --- Visualisation des Scores ---
score_sain_mean = df[df['TARGET_CLASS'] == 'F0SV0']['anomaly_score'].mean()
score_defaut_mean = df[df['TARGET_CLASS'] != 'F0SV0']['anomaly_score'].mean()

print(f"\nScore d'Anomalie Moyen:")
print(f"  - Vols Sains (F0SV0): {score_sain_mean:.4f} (Doit être proche de 0 ou positif)")
print(f"  - Vols Défectueux:   {score_defaut_mean:.4f} (Doit être significativement négatif)")

# --- Évaluation Quantitative ---

# Pour évaluer la performance binaire (Sain vs. Défaut), nous utilisons l'AUC-ROC.
# L'AUC (Area Under the Curve) mesure la capacité du modèle à séparer les deux classes.
# Un bon score d'anomalie produit un AUC proche de 1.0.

# L'AUC-ROC nécessite que les scores d'anomalie soient inversés (plus grand = défaut)
# Isolation Forest: score négatif = défaut. Donc, -score = défaut (positif)
auc_score = roc_auc_score(y_true_binary, -df['anomaly_score'])

print(f"\n--- Performance du Modèle (Sain vs. Défaut) ---")
print(f"AUC-ROC Score: {auc_score:.4f} (Objectif > 0.85)")

# Afficher les pires scores d'anomalie pour voir si ce sont bien des cas de défauts sévères
print("\n--- 5 Plus Grandes Anomalies Détectées (Score le Plus Négatif) ---")
print(df.sort_values(by='anomaly_score').head(5)[['FileName', 'TARGET_CLASS', 'anomaly_score']])