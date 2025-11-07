import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- 1. Configuration et Chargement des Données ---
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'transformed_dataset.csv')
df = pd.read_csv(file_path)

# LISTE COMPLÈTE des 12 caractéristiques V2
FEATURE_COLUMNS = [
    'Motor_Roll_Diff_Mean_DS', 'AccZ_IMU1_RMS_DS', 'AccZ_IMU1_Skewness_DS', 
    'Roll_Control_Error_Var', 'IMU_Roll_Diff_Std', 'IMU_Pitch_Diff_Std',
    'FFT_AccZ_Peak1_Amp', 'FFT_AccZ_Peak1_Freq', 'FFT_AccZ_Peak2_Amp', 
    'FFT_Energy_MotorBand', 'FFT_Energy_HarmonicBand', 'RefThrust_Mean_DS'
]

# Préparation des variables cibles et explicatives
df['TARGET_CLASS_SIMPLIFIED'] = df['TARGET_CLASS'].str[0:3] 
X = df[FEATURE_COLUMNS]
y_simplified = df['TARGET_CLASS_SIMPLIFIED']

# Séparation des données (stratifiée sur les 4 classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_simplified, test_size=0.25, random_state=42, stratify=y_simplified
)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. Entraînement du Modèle ---
# Re-entraînement rapide pour obtenir les prédictions (y_pred) et l'importance des features
rf_classifier = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42, 
    class_weight='balanced'
)
rf_classifier.fit(X_train_scaled, y_train)
y_pred = rf_classifier.predict(X_test_scaled)
classes = sorted(y_test.unique()) 


# --- 3. Visualisation 1 : Matrice de Confusion ---

def plot_confusion_matrix(y_true, y_pred, classes):
    """Génère et affiche la matrice de confusion normalisée par Recall."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Normalisation par ligne (Recall) pour voir le taux de succès réel par classe
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt=".2f", # Afficher 2 décimales pour les pourcentages de Recall
        cmap="Blues", 
        xticklabels=classes, 
        yticklabels=classes
    )
    plt.title('Matrice de Confusion (Normalisée par Recall)')
    plt.ylabel('Classe Réelle (True Label)')
    plt.xlabel('Classe Prédite (Predicted Label)')
    plt.show()

print("\nGénération de la Matrice de Confusion...")
plot_confusion_matrix(y_test, y_pred, classes)
#


# --- 4. Visualisation 2 : Importance des Caractéristiques ---

def plot_feature_importance(model, features):
    """Génère et affiche le diagramme à barres d'importance des caractéristiques."""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=feature_importance_df, 
        palette='viridis'
    )
    plt.title('Importance des Caractéristiques (Random Forest)')
    plt.xlabel('Score d\'Importance')
    plt.ylabel('Caractéristique')
    plt.tight_layout()
    plt.show()

print("\nGénération du Diagramme d'Importance des Caractéristiques...")
plot_feature_importance(rf_classifier, FEATURE_COLUMNS)
#