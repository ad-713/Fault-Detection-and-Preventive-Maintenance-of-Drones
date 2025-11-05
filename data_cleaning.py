import scipy.io
import pandas as pd
import numpy as np
import os

# --- EXPORTATION DU FICHIER .MAT EN .CSV DANS PYTHON ---

# Le chemin est désormais relatif au script qui s'exécute
file_path = 'Data'
# Le 'filename' est le nom exact du fichier .mat.
file_name = 'F0_SV0_SP1_t1_D1_R1.mat'

# Construction du chemin complet du fichier .mat
mat_file_path = os.path.join(file_path, file_name)

# 1. Charger le fichier .mat dans un dictionnaire Python
# Assurez-vous que le dossier 'Data' existe et contient le fichier .mat
mat_content = scipy.io.loadmat(mat_file_path)

# 2. Extraire la variable principale
QDrone_array = mat_content['QDrone_data']

# 3. Convertir en DataFrame Pandas
df = pd.DataFrame(QDrone_array)
# Transposition pour avoir les données (les observations) sur les lignes
df_to_export = df.T

# 4. Exporter en fichier CSV
# index=False pour omettre la colonne d'index (les numéros de ligne)
# header=False pour omettre les noms de colonnes (puisque la matrice est purement numérique)
csv_name = 'QDrone_data_python.csv'
# Chemin où le CSV sera enregistré (dans le dossier 'Data')
csv_path = os.path.join(file_path, csv_name)

df_to_export.to_csv(csv_path, index=False, header=False) 

print(f"✅ Exportation {csv_name} (Format Lignes) terminée et sauvegardée dans : {file_path}")

# --- VÉRIFICATION RAPIDE DU CSV DANS PYTHON ---

# 5. Charger le fichier CSV (pour vérifier le contenu)
df_check = pd.read_csv(csv_path, header=None)

print("\n** RÉSULTATS DE LA VÉRIFICATION **")
print("Taille du DataFrame (Lignes x Colonnes) après transposition :")
# DOIT afficher (87837, 56) si la transposition a fonctionné
print(df_check.shape) 

print("\nAperçu des 5 premières lignes et 5 premières colonnes :")
print(df_check.iloc[0:5, 0:5])