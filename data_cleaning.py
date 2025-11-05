import scipy.io
import pandas as pd
import numpy as np
import os

# --- EN-TÊTES DE COLONNES SPÉCIFIÉES (41 variables) ---
# Ces noms seront utilisés pour nommer les 41 colonnes conservées.
specified_column_names = [
    'Time', 
    # IMU #1 (9)
    'IMU1_Roll', 'IMU1_Pitch', 'IMU1_Yaw', 'IMU1_Roll_Rate', 'IMU1_Pitch_Rate',
    'IMU1_Yaw_Rate', 'IMU1_Roll_Accel', 'IMU1_Pitch_Accel', 'IMU1_Yaw_Accel',
    # IMU #2 (9)
    'IMU2_Roll', 'IMU2_Pitch', 'IMU2_Yaw', 'IMU2_Roll_Rate', 'IMU2_Pitch_Rate',
    'IMU2_Yaw_Rate', 'IMU2_Roll_Accel', 'IMU2_Pitch_Accel', 'IMU2_Yaw_Accel',
    # BATTERIE (1)
    'Battery_Level',
    # Gyro #1 & Accel #1 (6)
    'Gyro1_Roll_Rate', 'Gyro1_Pitch_Rate', 'Gyro1_Yaw_Rate', 
    'Accel1_X', 'Accel1_Y', 'Accel1_Z',
    # Gyro #2 & Accel #2 (6)
    'Gyro2_Roll_Rate', 'Gyro2_Pitch_Rate', 'Gyro2_Yaw_Rate', 
    'Accel2_X', 'Accel2_Y', 'Accel2_Z',
    # HAUTEUR (1)
    'Height_Range_Data',
    # COMMANDES MOTEURS (8)
    'FL_Motor_Cmd', 'FL_ESC_Cmd', 'FR_Motor_Cmd', 'FR_ESC_Cmd', 
    'BL_Motor_Cmd', 'BL_ESC_Cmd', 'BR_Motor_Cmd', 'BR_ESC_Cmd'
]
# Vérification de sécurité : 
assert len(specified_column_names) == 41, "Erreur: La liste d'en-têtes spécifiés doit contenir 41 noms."

# Mapping des indices des colonnes à conserver (basé sur votre tableau)
# Les indices de DataFrame/Python commencent à 0
# Indices 20, 21, 22, 23, 25, 26, 39 à 45, 55, 56 sont ceux à supprimer.
# Indices des colonnes à CONSERVER (0 à 55, hors indices non spécifiés)
columns_to_keep_indices = [
    0,                                                  # 1. Time (Index 0)
    1, 2, 3, 4, 5, 6, 7, 8, 9,                          # 2-10. IMU #1 (Indices 1 à 9)
    10, 11, 12, 13, 14, 15, 16, 17, 18,                 # 11-19. IMU #2 (Indices 10 à 18)
    23,                                                 # 24. Battery Level (Index 23)
    26, 27, 28, 29, 30, 31,                             # 27-32. Gyro #1 & Accel #1 (Indices 26 à 31)
    32, 33, 34, 35, 36, 37,                             # 33-38. Gyro #2 & Accel #2 (Indices 32 à 37)
    45,                                                 # 46. Height Range Data (Index 45)
    46, 47, 48, 49, 50, 51, 52, 53                      # 47-54. Commandes Moteurs (Indices 46 à 53)
]

# --- PARAMÈTRES ET CHEMINS ---

file_path = 'Data'
file_name = 'F0_SV0_SP1_t1_D1_R1.mat'
csv_name = 'QDrone_data.csv'

mat_file_path = os.path.join(file_path, file_name)
csv_path = os.path.join(file_path, csv_name)

# --- EXPORTATION ET FILTRAGE ---

# 1. Charger le fichier .mat
mat_content = scipy.io.loadmat(mat_file_path)
QDrone_array = mat_content['QDrone_data']

# 2. Convertir et Transposer (taille : 87837 x 56)
df = pd.DataFrame(QDrone_array)
df_transposed = df.T 

# 3. FILTRAGE : Conserver uniquement les 41 colonnes spécifiées
# Utilise .iloc pour sélectionner les colonnes par leurs indices
df_filtered = df_transposed.iloc[:, columns_to_keep_indices]

# 4. Appliquer les noms de colonnes clairs (41 noms)
df_filtered.columns = specified_column_names

# 5. Exporter en fichier CSV (avec en-têtes)
# L'écrasement est automatique.
df_filtered.to_csv(csv_path, index=False, header=True) 

print(f"✅ Exportation {csv_name} terminée et sauvegardée dans : {file_path}")

# --- VÉRIFICATION RAPIDE ---

# 6. Charger le fichier CSV
df_check = pd.read_csv(csv_path)

print("\n** RÉSULTATS DE LA VÉRIFICATION **")
print("Taille du DataFrame (Lignes x Colonnes) après filtrage :")
# DOIT afficher (87837, 41)
print(df_check.shape) 

print("\nAperçu des 5 premières lignes et des EN-TÊTES de colonnes :")
print(df_check.head())