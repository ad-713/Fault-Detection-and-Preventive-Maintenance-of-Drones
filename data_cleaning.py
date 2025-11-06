import scipy.io
import pandas as pd
import numpy as np
import os
import glob # Nécessaire pour lister les fichiers

# --- 1. CONSTANTES : NOMS ET INDICES DES COLONNES À CONSERVER (41) ---

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

# Indices des 41 colonnes à CONSERVER (0-basés sur les 56 colonnes originales)
columns_to_keep_indices = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
    10, 11, 12, 13, 14, 15, 16, 17, 18, 
    23, 
    26, 27, 28, 29, 30, 31, 
    32, 33, 34, 35, 36, 37, 
    45, 
    46, 47, 48, 49, 50, 51, 52, 53
]
assert len(columns_to_keep_indices) == 41, "Erreur dans le nombre d'indices."
assert len(specified_column_names) == 41, "Erreur dans le nombre de noms."


# --- 2. DÉFINITION DU DOSSIER CIBLE ---

base_path = 'Data'
sub_folder = 'DronePropA Motion Trajectories Dataset'
# Chemin complet du dossier contenant tous les fichiers .mat
full_data_directory = os.path.join(base_path, sub_folder)

# Utiliser glob pour trouver tous les fichiers *.mat
mat_files_list = glob.glob(os.path.join(full_data_directory, '*.mat'))

print(f"Dossier cible : {full_data_directory}")
print(f"Nombre de fichiers .mat trouvés : {len(mat_files_list)}\n")

# --- 3. BOUCLE DE TRAITEMENT PAR LOT ---

for mat_file_path in mat_files_list:
    
    # Extraire le nom du fichier (ex: F0_SV0_SP1_t1_D1_R1)
    file_name_with_ext = os.path.basename(mat_file_path)
    file_base_name = os.path.splitext(file_name_with_ext)[0]
    
    # Définir le chemin de sortie du CSV (dans le même dossier que le .mat)
    csv_name = f"{file_base_name}_QDrone.csv"
    csv_path = os.path.join(full_data_directory, csv_name)

    print(f"--- Traitement de : {file_name_with_ext} ---")

    try:
        # 3.1 Charger le fichier .mat
        mat_content = scipy.io.loadmat(mat_file_path)

        # 3.2 Extraire, convertir et transposer (taille : X x 56)
        QDrone_array = mat_content['QDrone_data']
        df = pd.DataFrame(QDrone_array)
        df_transposed = df.T 

        # 3.3 Filtrage et application des noms
        df_filtered = df_transposed.iloc[:, columns_to_keep_indices]
        df_filtered.columns = specified_column_names

        # 3.4 Exporter en CSV (header=True, écrasement automatique)
        df_filtered.to_csv(csv_path, index=False, header=True)
        
        print(f"  ✅ Succès : Enregistré sous {csv_name} (Taille : {df_filtered.shape})")

    except KeyError:
        print(f"  ❌ Erreur : La variable 'QDrone_data' est introuvable dans {file_name_with_ext}. Fichier ignoré.")
    except Exception as e:
        print(f"  ❌ Erreur inconnue lors du traitement de {file_name_with_ext}: {e}")

print("\nProcessus de conversion par lot terminé.")