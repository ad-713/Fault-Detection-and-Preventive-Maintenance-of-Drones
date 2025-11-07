import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# --- 1. Configuration et Chargement des Données ---
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'transformed_dataset.csv')
df = pd.read_csv(file_path)

# Caractéristiques à utiliser (basées sur le Top 10 d'importance)
FEATURE_COLUMNS = [
    'Motor_Roll_Diff_Mean_DS', 'AccZ_IMU1_RMS_DS', 'AccZ_IMU1_Skewness_DS', 
    'Roll_Control_Error_Var', 'IMU_Roll_Diff_Std', 'IMU_Pitch_Diff_Std',
    'FFT_AccZ_Peak1_Amp', 'FFT_AccZ_Peak1_Freq', 'FFT_AccZ_Peak2_Amp', 
    'FFT_Energy_MotorBand', 'FFT_Energy_HarmonicBand', 'RefThrust_Mean_DS'
]

# Variables cibles réelles (utilisées seulement pour l'évaluation post-clustering)
df['FAULT_TYPE_BASE'] = df['TARGET_CLASS'].str[0:3] # Ex: F0S, F1S, F2S, F3S

X = df[FEATURE_COLUMNS]
y_true = df['FAULT_TYPE_BASE'] # Étiquettes simplifiées pour l'analyse

# Définition du nombre de clusters (K=4 : Sain + F1 + F2 + F3)
K_CLUSTERS = 4 

print(f"Jeu de données chargé. Dimensions: {X.shape}")
print(f"Objectif de clustering fixé à K = {K_CLUSTERS} (Sain + 3 types de défauts).")

# --- 2. Standardisation des Caractéristiques ---

# La standardisation est cruciale pour K-Means car il utilise la distance Euclidienne
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Entraînement du Modèle K-Means ---

# random_state pour la reproductibilité
# n_init='auto' pour la version moderne de scikit-learn
kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_scaled)

# Ajout des labels de clusters au DataFrame original pour l'analyse
df['Cluster_Label'] = cluster_labels
df['Cluster_Center_Distance'] = kmeans.transform(X_scaled).min(axis=1) # Distance au centre de son cluster

print("\nClustering K-Means terminé.")

# --- 4. Évaluation et Interprétation des Clusters ---

# Métrique d'évaluation interne (cohérence du regroupement)
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Score de Silhouette (Indice de cohésion interne): {silhouette_avg:.4f}")

# Analyse de la composition des clusters (Interprétation)
print("\n--- Composition des Clusters (Vérité Terrain vs. Cluster) ---")

# Nous groupons par label de cluster et comptons la distribution des types de défauts réels
cluster_analysis = df.groupby('Cluster_Label')['FAULT_TYPE_BASE'].value_counts(normalize=False).unstack(fill_value=0)

# Pour chaque cluster, mettez en évidence la classe dominante
for i in range(K_CLUSTERS):
    total_samples = cluster_analysis.loc[i].sum()
    dominant_type = cluster_analysis.loc[i].idxmax()
    dominant_count = cluster_analysis.loc[i].max()
    purity = (dominant_count / total_samples) * 100
    
    print(f"\nCluster {i} (N={total_samples}):")
    print(f"   -> Dominance: {dominant_type} ({purity:.1f} % de pureté)")
    print(cluster_analysis.loc[i].sort_values(ascending=False))

# --- 5. Identification des Anomalies Potentielles (Cas Extrêmes) ---

# Les échantillons qui sont très éloignés du centre de leur propre cluster sont des anomalies
# Utile pour trouver les cas de défauts de haute gravité qui ne rentrent pas bien.
threshold = df['Cluster_Center_Distance'].quantile(0.95)
anomalies = df[df['Cluster_Center_Distance'] > threshold]

print(f"\n--- Anomalies (Top 5% des distances aux centres) ---")
print(f"Nombre d'anomalies potentielles (distance > {threshold:.2f}): {len(anomalies)}")
print("Exemples de classes d'anomalies (réelles) :")
print(anomalies['TARGET_CLASS'].value_counts())