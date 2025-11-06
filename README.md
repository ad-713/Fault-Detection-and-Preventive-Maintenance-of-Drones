# 📑 Document Méthodologique Final — *Feature Engineering* du Dataset **DronePropA**

L’étape de **Feature Engineering** a été affinée pour isoler les **signatures dynamiques et de commande** les plus discriminantes pour la classification des défauts.  
La stratégie a abouti à un jeu de **12 caractéristiques** (*features*) de haute valeur.  
Chaque vol est transformé en **une seule ligne de données** dans le jeu de données final.

---

## 🧩 1. Stratégie de Gestion des Données Brutes et Étiquettes

### A. Définition des Étiquettes de Sortie

| **Colonne** | **Rôle** |
| :-- | :-- |
| `TARGET_CLASS_SIMPLIFIED` | **Variable Cible Finale (4 Classes)** : `F0S` (Sain), `F1S` (Coupure de Bord), `F2S` (Fissure), `F3S` (Coupure de Surface). La gravité (`SV1–SV3`) est ignorée pour stabiliser le modèle. |

---

### B. Gestion de la Volumétrie (*Downsampling*)

| **Paramètre** | **Valeur** | **Rôle dans l’Extraction** |
| :-- | :-- | :-- |
| Fréquence d’Échantillonnage (FS) | **1000 Hz** | Fréquence native des capteurs. |
| Facteur de Downsampling | **10** | Un échantillon sur dix est conservé pour les calculs dans le domaine temporel (réduction de la charge CPU). |
| Exception Downsampling | **Signal `Acc_Z`** | Conservé à **1 kHz** pour l’analyse spectrale (FFT/Welch) afin de capturer les harmoniques de vibration critiques. |

---

## ⚙️ 2. Détails des Caractéristiques — *Tableau Synthétique (V2)*

Ce tableau récapitule les **12 caractéristiques finales** retenues pour le modèle de classification simplifiée.  
Elles couvrent trois domaines principaux : **Commande / Dynamique / Fréquentiel**.

| **Caractéristique** | **Domaine** | **Matrice Source (Ligne)** | **Calcul Détaillé** |
| :-- | :-- | :-- | :-- |
| `RefThrust_Mean_DS` | Commande | `commander_data (34)` | Moyenne de la série temporelle de Ref Thrust sur les données sous-échantillonnées (`_DS`). |
| `Motor_Roll_Diff_Mean_DS` | Commande | `QDrone_data (47, 49, 51, 53)` | Moyenne de la série temporelle du différentiel de commande : (Cmd FL + Cmd BL) − (Cmd FR + Cmd BR) sur les données sous-échantillonnées (`_DS`). |
| `Roll_Control_Error_Var` | Erreur | `QDrone_data (2)` et `commander_data (35)` | Variance de l’erreur de roulis : Variance(Ref Roll Angle − Measured Roll) sur les données sous-échantillonnées (`_DS`). |
| `AccZ_IMU1_RMS_DS` | Dynamique | `QDrone_data (32)` | RMS (Racine Carrée Moyenne) de la série temporelle de l’Acceleration along Z sur les données sous-échantillonnées (`_DS`). |
| `AccZ_IMU1_Skewness_DS` | Dynamique | `QDrone_data (32)` | Skewness (Asymétrie) de la série temporelle de l’Acceleration along Z sur les données sous-échantillonnées (`_DS`). |
| `IMU_Roll_Diff_Std` | Redondance | `QDrone_data (2, 11)` | Écart-type de la série temporelle de la différence entre : Measured Roll IMU1 − Measured Roll IMU2 sur les données sous-échantillonnées (`_DS`). |
| `IMU_Pitch_Diff_Std` | Redondance | `QDrone_data (3, 12)` | Écart-type de la série temporelle de la différence entre : Measured Pitch IMU1 − Measured Pitch IMU2 sur les données sous-échantillonnées (`_DS`). |
| `FFT_AccZ_Peak1_Amp` | Fréquentiel | `QDrone_data (32)` | Amplitude du pic le plus fort (Peak 1) de la Densité Spectrale de Puissance (DSP) obtenue via la méthode Welch. Calculé sur la série complète (`_FULL`). |
| `FFT_AccZ_Peak1_Freq` | Fréquentiel | `QDrone_data (32)` | Fréquence correspondant au pic le plus fort (Peak 1) de la DSP obtenue via la méthode Welch. Calculé sur la série complète (`_FULL`). |
| `FFT_AccZ_Peak2_Amp` | Fréquentiel | `QDrone_data (32)` | Amplitude du deuxième pic le plus fort (Peak 2) de la DSP obtenue via la méthode Welch. Calculé sur la série complète (`_FULL`). |
| `FFT_Energy_MotorBand` | Fréquentiel | `QDrone_data (32)` | Somme de la Puissance Spectrale (DSP) dans la bande de fréquence fondamentale du moteur (30 Hz → 150 Hz). Calculé sur la série complète (`_FULL`). |
| `FFT_Energy_HarmonicBand` | Fréquentiel | `QDrone_data (32)` | Somme de la Puissance Spectrale (DSP) dans la bande harmonique du moteur (150 Hz → 300 Hz). Calculé sur la série complète (`_FULL`). |

---

## 🧠 Résumé Méthodologique

- Les **features de commande** capturent l’effort de stabilisation du contrôleur (déséquilibres moteurs).  
- Les **features dynamiques** mesurent la réponse mécanique brute (IMU).  
- Les **features fréquentielles** isolent les harmoniques moteurs critiques responsables des vibrations caractéristiques des défauts.

> ✅ Ce jeu de 12 variables V2 a permis une **amélioration nette de la performance de classification** grâce à une meilleure robustesse face au bruit et à la redondance capteur.
