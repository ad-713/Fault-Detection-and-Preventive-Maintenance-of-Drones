import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Charger le dataset de features V2
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'transformed_dataset.csv')
df = pd.read_csv(file_path)

# NOUVELLE LISTE des 12 caractéristiques V2
FEATURE_COLUMNS = [
    'Motor_Roll_Diff_Mean_DS', 'AccZ_IMU1_RMS_DS', 'AccZ_IMU1_Skewness_DS', 
    'Roll_Control_Error_Var', 'IMU_Roll_Diff_Std', 'IMU_Pitch_Diff_Std',
    'FFT_AccZ_Peak1_Amp', 'FFT_AccZ_Peak1_Freq', 'FFT_AccZ_Peak2_Amp', 
    'FFT_Energy_MotorBand', 'FFT_Energy_HarmonicBand', 'RefThrust_Mean_DS'
]

# --- Création de la Variable Cible Simplifiée (4 classes) ---
# La classe cible est le Type de Défaut (F) + 'S' (pour Signatures)
df['TARGET_CLASS_SIMPLIFIED'] = df['TARGET_CLASS'].str[0:3] 
# Exemples : F0SV0, F0SV1, F0SV2, F0SV3 deviennent tous F0S, F1SV1/F1SV2/F1SV3 deviennent F1S, etc.

X = df[FEATURE_COLUMNS]
y_simplified = df['TARGET_CLASS_SIMPLIFIED']

print(f"Nouvelles classes cibles : {y_simplified.unique()}")
print(f"Distribution des classes : \n{y_simplified.value_counts()}")

# Séparation des données (stratifiée sur les 4 classes simplifiées)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_simplified, test_size=0.25, random_state=42, stratify=y_simplified
)

# Standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialisation du classifieur
rf_classifier = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42, 
    class_weight='balanced' # Essentiel car le nombre d'échantillons est limité
)

# Entraînement du modèle
print("\nDébut de l'entraînement du Random Forest pour 4 classes...")
rf_classifier.fit(X_train_scaled, y_train)
print("Entraînement terminé.")

# Prédictions et Rapport
y_pred = rf_classifier.predict(X_test_scaled)

print("\n--- Rapport de Classification (4 Classes Simplifiées) ---")
print(classification_report(y_test, y_pred))

# Afficher l'Accuracy pour une métrique simple
print(f"Accuracy totale (4 classes) : {accuracy_score(y_test, y_pred):.4f}")