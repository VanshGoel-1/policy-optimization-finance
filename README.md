# Policy Optimization for Financial Decision-Making

This project, completed for the Shodh AI take-home assignment, evaluates two machine learning approaches to optimize loan approvals using the LendingClub dataset.

1.  A **Predictive Deep Learning (DL) Model** was trained to predict the *probability of default*.
2.  A **Prescriptive Offline Reinforcement Learning (RL) Agent** was trained to learn a *policy* to maximize financial returns.

The results show that the RL agent, which is optimized for profit, dramatically outperforms the historical "approve-all" strategy.

## 🚀 Key Results

| Metric | Model 1: Deep Learning (DL) | Model 2: Offline RL (CQL) |
| :--- | :--- | :--- |
| **Primary Goal** | Predict Default Risk (Probability) | Maximize Financial Return (Policy) |
| **Key Metric** | **ROC AUC Score** | **Estimated Policy Value** |
| **Metric Value** | **0.7329** | **$212.50** (per loan) |
| F1-Score (Class 1) | 0.2348 | N/A |
| Baseline (Historical) | N/A | **$-1806.30** (per loan) |

---

## 1. Model 1: Predictive Deep Learning (DL) Analysis

This model was a Multi-Layer Perceptron (MLP) trained to predict a binary target: {0: Fully Paid, 1: Defaulted}.

### Explaining the Metrics

* **Why ROC AUC (0.7329)?** This is the primary metric for this model. It measures the model's ability to **rank** applicants. A score of 0.7329 means our model is significantly better than a random guess (0.5) at giving high-risk applicants a higher *probability score* than low-risk applicants. It's a good measure of its predictive power.

* **Why F1-Score (0.2348)?** This metric (for the "Default" class) is a balance of precision and recall. Our low F1-score is a direct result of the **extreme class imbalance** in the data (far more "Paid" loans than "Defaulted" loans). It indicates that to catch even a small number of defaults, the model incorrectly flags many good loans, making it an unreliable tool for a simple yes/no decision.

---

## 2. Model 2: Offline Reinforcement Learning (RL) Analysis

This model, a **Discrete Conservative Q-Learning (CQL) agent**, was trained to learn a *policy* (Approve/Deny) that maximizes a reward signal (profit/loss).

### Explaining the Metrics

* **Why Estimated Policy Value?** This is the ultimate business metric. It answers the question: "If we deploy this agent, how much money will we make or lose per loan?"
* **Result ($212.50):** The analysis shows that our new RL policy has an estimated value of **`_212.50` per loan**.
* **Comparison ($-1806.30):** This is a massive improvement over the historical baseline (an average *loss* of **`$-1806.30`** per loan), which represented the "approve-all" strategy present in our dataset. This proves the RL agent successfully learned a profitable policy to **deny high-loss, high-risk loans.**

---

## 3. Policy Comparison & Future Steps

### Why the Policies Differ
The DL model and the RL agent perform different tasks:
* The **DL Model** is a passive **predictor**. It only tells you *what* it thinks will happen (e.g., "70% chance of default").
* The **RL Agent** is an active **decision-maker**. It tells you *what to do* (Approve/Deny) to maximize profit.

The RL agent can make more nuanced, "reward-aware" decisions.

**Example 1: The High-Risk, High-Loss Loan**
* **Applicant:** High loan amount ($30,000) and a high interest rate (22%).
* **DL Model:** Predicts a **60% chance of default**. It flags this as "High Risk."
* **RL Agent:** It also sees the high risk. But it calculates the reward: a potential loss of `-$30,000` is catastrophic. To protect its average profit, the agent's policy is to **DENY** this loan.

**Example 2: The High-Risk, Low-Loss Loan (The "Reward" Hint)**
* **Applicant:** Very low loan amount ($1,000) but a very high interest rate (25%).
* **DL Model:** Predicts a **70% chance of default**. This is a very high-risk applicant.
* **RL Agent:** It sees the 70% risk, but also sees the reward. The potential loss is *only* `-$1,000`. The agent's policy may have learned that this is a **good gamble**. The small potential loss doesn't hurt the average, so it **APPROVES** the loan to chase the high-interest profit. This is a decision a simple DL predictor cannot make.

### Future Steps & Limitations
1.  **Limitations:** The reward function used was a simple proxy (interest vs. principal). A real-world model would need to account for the time value of money and partial repayments.
2.  **Future Data:** The model would be vastly improved by having data on *denied* loan applications to reduce the bias from only seeing "approved" loans.
3.  **Other Algorithms:** We could explore other, more advanced DL models (like Transformers) or RL algorithms (like DDQN) to see if they can find an even more profitable policy.

---

## 🛠️ Setup

1.  Install **Miniconda** (for Python 3.10) and **Git**.
2.  Clone this repository:
    ```bash
    git clone [https://github.com/](https://github.com/)VanshGoel-1/policy-optimization-finance.git
    cd policy-optimization-finance
    ```
3.  Create and activate the Conda environment:
    ```bash
    # We use Conda to handle the complex dependencies
    conda create --name shodh_env python=3.10
    conda activate shodh_env
    ```
4.  Install all required packages (this uses the unified Conda install command):
    ```bash
    conda install -c pytorch -c conda-forge pytorch torchvision cpuonly pandas scikit-learn matplotlib seaborn jupyterlab pyarrow d3rlpy kaggle tqdm joblib ipykernel
    ```

##  reproducing the Results

1.  **Download Data:**
    * Create a Kaggle account and download your `kaggle.json` API token.
    * Place `kaggle.json` in `C:\Users\[Your-Username]\.kaggle\` (Windows) or `~/.kaggle/` (macOS/Linux).
    * Run the download script (or download manually to `data/raw/`):
        ```bash
        kaggle datasets download -d mparchnever/lending-club-loan-data -f accepted_2007_to_2018Q4.csv.gz -p data/raw/
        # Unzip the file
        ```
2.  **Run Notebooks in Order:**
    * Start the Jupyter environment:
        ```bash
        jupyter lab
        ```
    * Open and run the cells in `notebooks/01_EDA.ipynb`.
    * Open and run the cells in `notebooks/02_supervised.ipynb`.
    * Open and run the cells in `notebooks/03_offline_rl.ipynb`.
    * The `04_analysis.ipynb` notebook contains the final comparisons.

## 📞 Contact
* **Name:** Vansh Goel
* **GitHub:** `https://github.com/VanshGoel-1`