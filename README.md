## **GRAIL: Graph-based Active Learning on Dynamic Networks**

This repository provides the complete implementation of **GRAIL**, a benchmark for evaluating Active Learning (AL) strategies on dynamic graph-based data. GRAIL is designed to help researchers explore the performance of various AL strategies across different datasets, focusing on predictive performance, sampling diversity, and user burden.

---

### **Project Structure**

```
📁 Maryam-5112/
│
├── 📁 Dataset/
│   └── (Contains the dynamic graph datasets used for the experiments)
│
├── 📄 AL_strategies.py
│   └── Defines various Active Learning (AL) strategies used in the experiments.
│
├── 📄 Dataset_Description.md
│   └── Description of the datasets used in this project (FnF, SNAPSHOT).
│
├── 📄 Diversity_and_Burden_vs_topology_analysis.ipynb
│   └── Notebook for analyzing diversity and user burden metrics against network topology.
│
├── 📁 FnF_Results(intermediate).zip
│   └── Compressed folder containing intermediate results for FnF experiments.
│
├── 📄 Main_FnF_Model_run.ipynb
│   └── Main notebook for running the Active Learning (AL) experiments on FnF dataset.
│
├── 📄 Novel_Metrics_and_Plotting_utils.py
│   └── Utility functions for calculating custom metrics and creating plots.
│
├── 📄 Performance_Analysis_FnF.ipynb
│   └── Notebook for detailed performance analysis of AL strategies on FnF dataset.
│
├── 📄 Performance_Analysis_SNAPSHOT.ipynb
│   └── Notebook for detailed performance analysis of AL strategies on SNAPSHOT dataset.
│
├── 📄 Performance_vs_burden_tradeoff_analysis.ipynb
│   └── Notebook for analyzing the trade-off between performance and user burden.
│
├── 📄 Run_all_AL_FnF.py
│   └── Script for running all Active Learning (AL) strategies on FnF dataset.
│
├── 📄 Run_all_AL_SNAPSHOT.py
│   └── Script for running all Active Learning (AL) strategies on SNAPSHOT dataset.
│
├── 📄 Stream_Bootstrap_helpers.py
│   └── Helper functions for managing bootstrapping and stream processing.
│
├── 📄 data_loader.py
│   └── Data loading and preprocessing functions for the datasets.
│
├── 📄 metrics.py
│   └── Calculation of standard and custom metrics for evaluating AL performance.
│
├── 📄 models.py
│   └── Model definitions and training functions for the AL experiments.
│
└── 📄 README.md
    └── (This file)
```

---

### **Getting Started**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Maryam-5112.git
   cd Maryam-5112
   ```

2. **Install Required Packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Experiments:**

   * For FnF dataset:

     ```bash
     python Run_all_AL_FnF.py
     ```
   * For SNAPSHOT dataset:

     ```bash
     python Run_all_AL_SNAPSHOT.py
     ```

---

### **Key Features**

* Multiple Active Learning Strategies: Evaluate and compare a range of AL strategies (e.g., Random, GraphPart, AGE).
* Custom Metrics: Includes Cumulative Performance Index (CPI), Sampling Entropy, Coverage Ratio, and Time-Gap Analysis.
* Scalable: Supports bootstrapping and time-series evaluation for dynamic graph data.
* Visual Analysis: Ready-to-use notebooks for performance, diversity, and burden analysis.

---

### **Usage Notes**

* Datasets must be placed in the `Dataset/` directory.
* Modify the configuration (e.g., number of bootstraps, query size) in the script files as needed.
* Results are saved in the corresponding metrics folders for each dataset.

---
