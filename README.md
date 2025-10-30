Here is the content for your final `README.md` file, combining the project overview, key results, and submission instructions. You can copy and paste this directly into your file.

-----

# 💰 Policy Optimization for Financial Decision-Making

This project evaluates the business potential of two machine learning models for optimizing loan approval decisions using the LendingClub dataset.

The central finding is that the **Prescriptive Offline Reinforcement Learning (RL) Agent** learns a policy to maximize profit, fundamentally outperforming the traditional predictive model.

-----

## 🚀 Key Project Results

| Metric | Model 1: Deep Learning (DL) | Model 2: Offline RL (CQL) |
| :--- | :--- | :--- |
| **Primary Goal** | Predict Default Risk (Probability) | Maximize Financial Return (Policy) |
| **Key Metric** | **ROC AUC Score** | **Estimated Policy Value** |
| **Metric Value** | **0.7329** | **$212.50** (per loan) |
| F1-Score (Class 1) | 0.2348 | N/A |
| Baseline (Historical) | N/A | **$-1806.30** (per loan loss) |

-----

## 🧠 Model Comparison Insight

  * The **DL Model** acts as a **risk predictor**. It accurately ranks applicants (AUC 0.7329) but struggles with the class imbalance (low F1-Score), making it unreliable for simple binary decisions.
  * The **RL Agent** acts as a **profit maximizer**. It learns to strategically **Deny** loans with small principal but catastrophic loss potential, shifting the average expected return from a **$-\$1806.30$ loss** to a **$+\$212.50$ profit**. This demonstrates the superiority of a value-based policy over a probability-based predictor.

-----

## 🛠️ Project Setup & Reproduction

### Prerequisites

  * Python 3.10+
  * Miniconda / Conda
  * Git

### Repository Structure

```
policy-optimization-finance/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/ (Contains: accepted_2007_to_2018Q4.csv)
│  └─ processed/ (Contains: train.parquet, test.parquet)
├─ notebooks/
│  ├─ 01_EDA.ipynb
│  ├─ 02_supervised.ipynb
│  └─ 03_offline_rl.ipynb
├─ models/
│  └─ scaler.joblib
```

### Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone [Paste Your Repo URL Here]
    cd policy-optimization-finance
    ```
2.  **Create & Activate Environment:**
    ```bash
    conda create --name shodh_env python=3.10
    conda activate shodh_env
    ```
3.  **Install Dependencies:** (This uses a unified install to resolve PyTorch/d3rlpy conflicts)
    ```bash
    conda install -c pytorch -c conda-forge pytorch torchvision cpuonly pandas scikit-learn matplotlib seaborn jupyterlab pyarrow d3rlpy kaggle tqdm joblib ipykernel
    ```
4.  **Download Data:** Place the `accepted_2007_to_2018Q4.csv` file into the `data/raw/` folder.

### Run Experiment

1.  Start Jupyter Lab:
    ```bash
    jupyter lab
    ```
2.  Run the notebooks sequentially: `01_EDA.ipynb` $\rightarrow$ `02_supervised.ipynb` $\rightarrow$ `03_offline_rl.ipynb`.
3.  The final analysis and discussion are documented in the attached PDF report.

-----

## 📞 Contact

  * **Name:** Vansh Goel
  * **Email:** goelvansh424@gmail.com
