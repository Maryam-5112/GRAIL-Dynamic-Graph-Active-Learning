### README for GRAIL Dataset (SNAPSHOT and FnF)

## **Dataset Overview**

The GRAIL dataset consists of two dynamic social network datasets: **SNAPSHOT** and **FnF (Friends and Family)**. These datasets are part of the GRAIL benchmark for evaluating graph-based Active Learning (AL) strategies in dynamic sensing environments. The datasets are designed for research in graph neural networks (GNNs), dynamic graph learning, and Active Learning applications.

## **Dataset Structure**

The dataset is organized into two main folders:

* **`SNAPSHOT/`**: Contains the SNAPSHOT dataset, which includes processed multimodal sensor data from a longitudinal study of 200+ participants.
* **`FnF/` (Friends and Family)**: Contains the FnF dataset with multimodal sensor data collected from 130 participants, capturing their social interactions and communication patterns.

### **Folder Structure:**

```
Dataset/
├── SNAPSHOT/
│   └── SNAPSHOT_Processed_all_Cohort.pkl  # Processed GNN data for SNAPSHOT (Pickle File)
└── FnF/
    ├── L_*_*.npy        # Adjacency matrices (friends, calls, duration, couples, non-zero)
    ├── X_*.npy          # Feature matrices for different labels
    └── y_*.npy          # Label arrays for different labels
```

## **Data Description**

### **1. SNAPSHOT Dataset:**

* **Data Type:** Longitudinal, multimodal sensor data (wrist-worn sensors, phone metadata, and daily surveys).
* **Participants:** 200+ (158 female, 92 male) with an average age of 21 years.
* **File Format:** The data is provided as a pickle file (`SNAPSHOT_Processed_all_Cohort.pkl`).
* **Data Structure:** The file is a dictionary where:

  * **Keys:** Cohort IDs (e.g., Cohort 1, Cohort 2).
  * **Values:** A dictionary containing:

    * `X`: Feature tensor (days x users x features).
    * `y`: Label tensor (days x users x 1).
    * `A`: Adjacency matrix (users x users), representing social connections (SMS graph).

### **2. FnF (Friends and Family) Dataset:**

* **Data Type:** Multimodal sensor data from a community of 130 participants.
* **Features:**

  * `X`: Feature matrix (samples x features) for various labels (e.g., mood, sleep).
  * `y`: Label array (samples).
  * `L_*`: Adjacency matrices representing various social interactions:

    * `L_friend_*`: Friendship network.
    * `L_calls_*`: Call network.
    * `L_duration_*`: Call duration network.
    * `L_couple_*`: Network of couples.
    * `L_non_zero_*`: Non-zero interaction network.

## **Labels**

The datasets provide multiple labels (target variables), including:

* **`sleep_class`**: Binary classification of sleep duration (0: ≤6 hours, 1: >6 hours).
* **`mood` labels**: Multiple binary labels representing different mood states:

  * `angry_or_frustrated`
  * `calm_or_peaceful`
  * `sad_or_depressed`
  * `stressed_or_anxious`

