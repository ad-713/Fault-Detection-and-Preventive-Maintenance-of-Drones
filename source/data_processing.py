import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# --- Configuration (À adapter) ---
DATA_DIR = './Data/DronePropA Motion Trajectories Dataset/'
OUTPUT_FILE = 'transformed_dataset.csv'

# Fréquence d'échantillonnage (fixée selon l'article)
FS = 1000 # 1 kHz 

# Paramètre de Réduction de Données (Downsampling)
# Nous allons prendre seulement 1 point sur N pour les calculs statistiques/différentiels 
# pour économiser du temps CPU et de la mémoire lors du chargement des matrices.
# Un facteur de 10 est souvent un bon compromis (1000 Hz -> 100 Hz effectif).
DOWNSAMPLE_FACTOR = 10 

# Définition des indices de ligne (Row Index) pour chaque variable clé (voir Table 1)
# Note: Ces indices sont corrects pour le format .mat
INDICES = {
    # Commander_data (Refs et Mesures de position/angle)
    'Ref_Thrust': 34 - 1,
    'Ref_Roll_Angle': 35 - 1, # Nouveau: Angle de roulis commandé
    'Ref_Pitch_Angle': 36 - 1, # Nouveau: Angle de tangage commandé
    # QDrone_data (Dynamique)
    'Roll_IMU1': 2 - 1,
    'Pitch_IMU1': 3 - 1,
    'Yaw_IMU1': 4 - 1,
    'Roll_IMU2': 11 - 1, # Nouveau: IMU 2
    'Pitch_IMU2': 12 - 1, # Nouveau: IMU 2
    'Yaw_IMU2': 13 - 1, # Nouveau: IMU 2
    'Roll_Rate_IMU1': 5 - 1,
    'Acc_Z_Gyro1': 32 - 1,
    # Commandes Moteur (QDrone_data)
    'Motor_FL': 47 - 1, 
    'Motor_FR': 49 - 1,
    'Motor_BL': 51 - 1,
    'Motor_BR': 53 - 1,
}

all_features = []

def extract_features_from_file_v2(filepath):
    """Charge un fichier .mat et extrait les caractéristiques V2 (avancées)."""
    
    try:
        data = loadmat(filepath)
    except Exception as e:
        print(f"Erreur de chargement du fichier {filepath}: {e}")
        return None

    commander_data = data['commander_data']
    qdrone_data = data['QDrone_data']
    features = {}
    
    # Données Sous-échantillonnées (DS)
    commander_ds = commander_data[:, ::DOWNSAMPLE_FACTOR]
    qdrone_ds = qdrone_data[:, ::DOWNSAMPLE_FACTOR]
    
    # Données Complètes (FULL) pour FFT
    acc_z_full = qdrone_data[INDICES['Acc_Z_Gyro1'], :]
    
    # --- A. Caractéristiques de Compensation et de Commande (V1 + Erreur) ---
    
    # Commandes moteurs (DS)
    cmd_fl = qdrone_ds[INDICES['Motor_FL'], :]
    cmd_fr = qdrone_ds[INDICES['Motor_FR'], :]
    cmd_bl = qdrone_ds[INDICES['Motor_BL'], :]
    cmd_br = qdrone_ds[INDICES['Motor_BR'], :]
    
    # V1 Feature: RefThrust_Mean_DS (CORRECTION CLÉ : AJOUTÉ ICI)
    ref_thrust_ds = commander_ds[INDICES['Ref_Thrust'], :]
    features['RefThrust_Mean_DS'] = np.mean(ref_thrust_ds) 
    
    # V1 Feature: Motor Roll Diff
    motor_roll_diff = (cmd_fl + cmd_bl) - (cmd_fr + cmd_br)
    features['Motor_Roll_Diff_Mean_DS'] = np.mean(motor_roll_diff)

    # V2 Feature: Erreur de Contrôle de Roulis (Variance de l'effort)
    measured_roll_ds = qdrone_ds[INDICES['Roll_IMU1'], :]
    ref_roll_ds = commander_ds[INDICES['Ref_Roll_Angle'], :]
    
    roll_error = ref_roll_ds - measured_roll_ds
    features['Roll_Control_Error_Var'] = np.var(roll_error) 
    
    # V1 Feature: Accélération Z
    acc_z_ds = qdrone_ds[INDICES['Acc_Z_Gyro1'], :]
    features['AccZ_IMU1_RMS_DS'] = np.sqrt(np.mean(acc_z_ds**2))
    features['AccZ_IMU1_Skewness_DS'] = skew(acc_z_ds)
    
    # --- B. Caractéristiques Différentielles (Redondance IMU) ---
    
    roll_imu1_ds = qdrone_ds[INDICES['Roll_IMU1'], :]
    roll_imu2_ds = qdrone_ds[INDICES['Roll_IMU2'], :]
    pitch_imu1_ds = qdrone_ds[INDICES['Pitch_IMU1'], :]
    pitch_imu2_ds = qdrone_ds[INDICES['Pitch_IMU2'], :]
    
    # V2 Feature: Écart-type des différences entre les IMU
    features['IMU_Roll_Diff_Std'] = np.std(roll_imu1_ds - roll_imu2_ds)
    features['IMU_Pitch_Diff_Std'] = np.std(pitch_imu1_ds - pitch_imu2_ds)
    
    # --- C. Caractéristiques Fréquentielles Ciblées (Toute la puissance) ---
    
    f, Pxx = welch(acc_z_full, FS, nperseg=256) 
    
    # 1. Fréquence d'Énergie Totale (V1, corrigée si nécessaire)
    valid_indices = f > 1
    f_valid = f[valid_indices]
    Pxx_valid = Pxx[valid_indices]
    
    if len(Pxx_valid) > 0:
        peak_indices = np.argsort(Pxx_valid)[-3:][::-1]
        features['FFT_AccZ_Peak1_Amp'] = Pxx_valid[peak_indices[0]] if len(peak_indices) > 0 else 0.0
        features['FFT_AccZ_Peak1_Freq'] = f_valid[peak_indices[0]] if len(peak_indices) > 0 else 0.0
        features['FFT_AccZ_Peak2_Amp'] = Pxx_valid[peak_indices[1]] if len(peak_indices) > 1 else 0.0
    else:
        features['FFT_AccZ_Peak1_Amp'] = features['FFT_AccZ_Peak1_Freq'] = features['FFT_AccZ_Peak2_Amp'] = 0.0

    # 2. V2 Feature: Énergie Fréquentielle Ciblée
    f_motor_start, f_motor_end = 30, 150 
    f_harm_start, f_harm_end = 150, 300 
    
    band_motor_indices = (f >= f_motor_start) & (f <= f_motor_end)
    features['FFT_Energy_MotorBand'] = np.sum(Pxx[band_motor_indices])
    
    band_harm_indices = (f >= f_harm_start) & (f <= f_harm_end)
    features['FFT_Energy_HarmonicBand'] = np.sum(Pxx[band_harm_indices])
    
    return features
# --- Boucle Principale de Traitement ---

print(f"Démarrage de l'extraction des caractéristiques dans {DATA_DIR}...")
print(f"Downsampling appliqué: 1 sur {DOWNSAMPLE_FACTOR} pour le domaine temporel.")
file_count = 0

for filename in os.listdir(DATA_DIR):
    if filename.endswith('.mat'):
        file_count += 1
        filepath = os.path.join(DATA_DIR, filename)
        
        # Extraction robuste des étiquettes (Labels)
        parts = filename.split('_')
        
        # Le nom de fichier minimum est F#_SV#_SP#_t#.mat
        if len(parts) >= 4:
            try:
                label_features = {
                    'FileName': filename,
                    # F#n: Fault group #n (0 pour sain)
                    'Fault_Type': int(parts[0][1:]), 
                    # SV#n: Severity level #n (0 pour sain)
                    'Severity_Level': int(parts[1][2:]),
                    # SP#n: Speed #n
                    'Speed_Level': int(parts[2][2:]),
                    # t#n: Trajectory #n
                    'Trajectory': int(parts[3].split('.')[0][1:]),
                }
            except ValueError:
                 # Gère les cas F0_SV0_SP1_t1_D1_R1.mat où F0 signifie sain
                 if parts[0] == 'F0' and parts[1] == 'SV0':
                     label_features = {
                        'FileName': filename,
                        'Fault_Type': 0, 
                        'Severity_Level': 0,
                        'Speed_Level': int(parts[2][2:]),
                        'Trajectory': int(parts[3].split('.')[0][1:]),
                     }
                 else:
                    print(f"Nom de fichier mal formaté ignoré (erreur de conversion) : {filename}")
                    continue
        else:
            print(f"Nom de fichier non standard ignoré : {filename}")
            continue

        ts_features = extract_features_from_file_v2(filepath)
        
        if ts_features:
            final_row = {**label_features, **ts_features}
            all_features.append(final_row)
            print(f"[{file_count}] Extrait : {filename}")
        
print("Extraction terminée.")

# --- Construction et Sauvegarde du DataFrame Final ---

if all_features:
    df_features = pd.DataFrame(all_features)
    
    # Création de la variable cible de classification détaillée
    # F0SV0 = Sain, F1SV1 = Edge Cut Level 1, etc.
    df_features['TARGET_CLASS'] = 'F' + df_features['Fault_Type'].astype(str) + 'SV' + df_features['Severity_Level'].astype(str)
    
    df_features.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Dataset final enregistré sous : {OUTPUT_FILE}")
    print(f"Dimensions du DataFrame : {df_features.shape} (Vols x Caractéristiques)")
else:
    print("\n⚠️ Aucune caractéristique n'a été extraite. Vérifiez le chemin d'accès aux données.")