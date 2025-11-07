# Hackathon BI Pipeline - Track UAV - Team 24
Gabriel GERMAIN
Adrien GREVET
Martin LAURENT
Alexandre HERVÉ
Daphné MARTY
Hugo LOUBIGNAC

# Document Méthodologique Final

---

# *Feature Engineering* du Dataset **DronePropA**

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

## Feature Importance

![Figure 1](./data%20display%20exemple/Plot/Feature_Importance.png)

**Analyse**

1. Compensation (Commande) : Motor_Roll_Diff_Mean_DS est la caractéristique la plus importante (Score ≈0.20). Cela prouve que l'effort de compensation asymétrique du pilote automatique pour contrer le déséquilibre de l'hélice est l'indicateur principal du défaut.

2. Vibrations Ciblées (Fréquentiel V2) : FFT_Energy_HarmonicBand (Énergie dans la bande 2× la fréquence moteur) est la deuxième caractéristique la plus importante (Score ≈0.12). Ce score très élevé valide l'efficacité de l'analyse fréquentielle ciblée par rapport à la simple analyse statistique.

3. Amplitude Vibratoire (Dynamique) : AccZ_IMU1_RMS_DS (Amplitude globale des vibrations) reste dans le top 3 (Score ≈0.13), confirmant que la force du bruit est un indicateur fondamental

---

## 🧠 Résumé Méthodologique

- Les **features de commande** capturent l’effort de stabilisation du contrôleur (déséquilibres moteurs).  
- Les **features dynamiques** mesurent la réponse mécanique brute (IMU).  
- Les **features fréquentielles** isolent les harmoniques moteurs critiques responsables des vibrations caractéristiques des défauts.

> ✅ Ce jeu de 12 variables V2 a permis une **amélioration nette de la performance de classification** grâce à une meilleure robustesse face au bruit et à la redondance capteur.

--- 

# 🚀 Analyse du Modèle Random Forest (Classification Simplifiée K=4)

L'entraînement du Random Forest sur les **4 classes simplifiées** (F0S, F1S, F2S, F3S) en utilisant les 12 caractéristiques V2 a permis une amélioration significative par rapport aux tentatives précédentes (Accuracy totale de 72.73% contre 45% pour les 10 classes).

---

## 1. Statistiques Globales et Validation de la Stratégie

| Métrique | Valeur | Interprétation |
|:---|:---|:---|
| **Accuracy** | **0.73** | Le modèle classe correctement 73% des vols. C'est un score élevé qui justifie l'approche par *feature engineering* avancée et la simplification des classes. |
| **Macro Avg F1-Score** | **0.71** | Solide. Le modèle est capable de diagnostiquer les quatre types de classes avec une bonne fiabilité, en tenant compte de la performance de chaque classe. |
| **Classes Entraînement** | F0S: 40, F1S: 30, F2S: 30, F3S: 30 | Les classes sont bien équilibrées, ce qui a stabilisé l'entraînement. |
| **Classes Test (Support)** | 10, 7, 8, 8 | Le faible support par classe de test (7 à 10 échantillons) limite toujours la performance absolue, mais les résultats sont robustes. |

---

## 2. Analyse Détaillée de la Performance par Classe

L'analyse montre que le modèle excelle à identifier la classe saine et a une bonne capacité de détection des défauts de type F2 (Fissure).

| Classe | Définition | Support | Precision | Recall | F1-Score | Interprétation (Sécurité/Diagnostic) |
|:---|:---|:---|:---|:---|:---|:---|
| **F0S** | **Sain** | 10 | **0.90** | **0.90** | **0.90** | **Excellent.** Le modèle est très fiable pour identifier un drone sain, minimisant les fausses alarmes. |
| **F2S** | **Fissure** | 8 | 0.58 | **0.88** | **0.70** | **Meilleur Recall (Détection).** Le modèle détecte 88% des Fissures. Le risque est la *faible Précision* (58%) : il confond F2S avec d'autres défauts (F1S, F3S) environ 42% du temps. |
| **F3S** | **Coupure de Surface** | 8 | **0.80** | 0.50 | 0.62 | **Meilleure Précision.** Lorsqu'il prédit F3S, il est correct 80% du temps. Le problème est le *faible Recall* (50%) : il manque la moitié des vrais F3S (qui sont classés comme F0S ou F2S). |
| **F1S** | **Coupure de Bord** | 7 | 0.67 | 0.57 | 0.62 | **Modéré.** Performance acceptable. Il confond les coupures de bord avec d'autres types de défauts (Precision) et manque certains cas (Recall). |

---

## Matrice de Confusion

![Figure 2](./data%20display%20exemple/Plot/Matrice_Confusion.png)

**Description**

La matrice de confusion, normalisée par le Recall (sensibilité), montre la proportion d'échantillons réels de chaque classe qui ont été correctement ou incorrectement prédits. La diagonale représente les taux de succès par classe.

**Analyse**

- Performance du Sain (F0S) : Le modèle excelle à identifier la classe saine, avec un Recall de 0.90. Seuls 10% des vols sains sont à tort classés comme F2S (Fissure), ce qui est un excellent résultat pour minimiser les fausses alarmes.

- Détection des Fissures (F2S) : La classe F2S a le meilleur Recall (0.88) parmi les défauts. Cela signifie que 88% des vrais défauts de type Fissure (F2S) sont correctement détectés. C'est la signature de défaut la plus distincte.

- Confusion F1S : La classe F1S (Coupure de Bord) a un faible Recall (0.57). Les erreurs se répartissent : 14% sont classés Sain (F0S) et 29% sont classés F2S. Le défaut F1S est majoritairement confondu avec le défaut F2S.

- Confusion F3S : La classe F3S (Coupure de Surface) a un faible Recall (0.50). Elle est manquée la moitié du temps. Les erreurs principales sont la confusion avec F1S (25%) et F2S (25%).

---

## 3. Conclusion et Stratégie d'Optimisation

Le modèle Random Forest est désormais un outil de diagnostic fonctionnel.

### Points Forts
1.  **Fiabilité F0S :** Le modèle excelle à déterminer si le drone est sain (F1-Score 0.90).
2.  **Détection F2S :** La signature des fissures (F2S) est très bien capturée (Recall 0.88).

### Point Faible (Confusion)
Le modèle présente une **confusion significative** entre les défauts de type F1S, F2S et F3S, comme en témoignent les F1-Scores modérés pour ces classes (0.62 à 0.70). Cette confusion est probablement due au fait que les défauts de faible gravité dans les trois groupes peuvent avoir des signatures dynamiques très similaires.

### Prochaine Étape Recommandée
Afin d'améliorer la Précision et le Recall pour les défauts F1S, F2S, et F3S, la prochaine étape logique est l'**Optimisation des Hyperparamètres du Random Forest**. L'ajustement du `max_depth` (profondeur maximale) et du `n_estimators` (nombre d'arbres) permettra au modèle de mieux exploiter les caractéristiques fines sans surapprendre les données.

Pour améliorer le projet, l'étape logique suivante consiste à prédire le **niveau de sévérité** en utilisant une **approche de diagnostic en cascade**. Cela implique d'entraîner des **modèles de spécialisation distincts** pour chaque type de défaut (F1S, F2S, F3S) afin de classifier la sévérité (SV1, SV2, SV3) avec une plus grande précision.