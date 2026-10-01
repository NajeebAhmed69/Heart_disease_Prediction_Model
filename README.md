```markdown
# 🫀 CardioCare: Heart Disease Risk Assessment & Classification

An end-to-end clinical machine learning project that benchmarks 9 classification algorithms, builds an automated diagnostic pipeline using **Scikit-Learn**, and deploys an interactive risk-scoring dashboard with **Streamlit**.

---

## 📌 Project Overview

Clinical decision-making requires models that generalize reliably, handle missing laboratory measurements without data leakage, and prioritize **Recall (Sensitivity)** alongside **ROC-AUC** to minimize potentially fatal False Negatives. 

This project implements:
- Automated missing-data imputation and feature encoding using Scikit-Learn `ColumnTransformer` and `Pipeline`.
- A 5-fold stratified cross-validation tournament across 9 supervised learning models.
- An interactive Streamlit web dashboard providing real-time risk classification and probability scoring.

---

## 📊 Dataset & Clinical Attributes

The model is trained on the **UCI Heart Disease (Cleveland)** dataset (303 patient records, 13 clinical predictors):

| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` | Numeric | Patient age in years |
| `sex` | Categorical | Biological sex (`1` = Male, `0` = Female) |
| `cp` | Categorical | Chest pain type (`1`: Typical angina, `2`: Atypical, `3`: Non-anginal, `4`: Asymptomatic) |
| `trestbps` | Numeric | Resting blood pressure (mm Hg on admission) |
| `chol` | Numeric | Serum cholesterol in mg/dl |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl (`1` = True, `0` = False) |
| `restecg` | Categorical | Resting ECG (`0`: Normal, `1`: ST-T wave abnormality, `2`: Left ventricular hypertrophy) |
| `thalach` | Numeric | Maximum heart rate achieved during exercise |
| `exang` | Categorical | Exercise-induced angina (`1` = Yes, `0` = No) |
| `oldpeak` | Numeric | ST depression induced by exercise relative to rest |
| `slope` | Categorical | Slope of the peak exercise ST segment (`1`: Upsloping, `2`: Flat, `3`: Downsloping) |
| `ca` | Categorical | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Categorical | Thallium heart scan (`3`: Normal, `6`: Fixed defect, `7`: Reversible defect) |
| **`target`** | **Binary Label** | **Heart disease diagnosis (`0` = Absence, `1` = Presence)** |

---

## ⚙️ Data Preprocessing & Leakage Prevention

All transformations are isolated inside Scikit-Learn `Pipeline` steps so that statistical parameters (medians, modes, scaling factors) are learned **strictly** on training folds:

- **Continuous Features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`):**
  - Missing values imputed using `SimpleImputer(strategy='median')` to prevent outlier distortion.
  - Normalized using `StandardScaler()`.
- **Categorical Features (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`):**
  - Missing values imputed using `SimpleImputer(strategy='most_frequent')`.
  - Encoded using `OneHotEncoder(drop='first', handle_unknown='ignore')` to eliminate multicollinearity.

---

## 🏆 Model Benchmarking & Results

Each classifier was evaluated using **5-Fold Stratified Cross-Validation**:

| Model | Accuracy | Recall (Sensitivity) | Precision | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **85.46%** | **79.76%** | **87.43%** | **83.21%** | **0.9164** |
| **Support Vector Machine (RBF)** | 82.83% | 79.13% | 83.00% | 80.89% | 0.8916 |
| **K-Nearest Neighbors (k=5)** | 82.19% | 78.44% | 82.04% | 80.14% | 0.8632 |
| **Gradient Boosting** | 79.55% | 76.96% | 78.71% | 77.60% | 0.8940 |
| **AdaBoost** | 80.18% | 76.14% | 79.62% | 77.62% | 0.8939 |
| **Extra Trees** | 82.52% | 75.53% | 85.58% | 79.97% | 0.9001 |
| **Random Forest** | 81.20% | 72.62% | 84.66% | 78.04% | 0.9056 |
| **Gaussian Naive Bayes** | 80.19% | 71.80% | 82.69% | 76.42% | 0.8841 |
| **Decision Tree (max_depth=4)** | 73.25% | 63.23% | 75.47% | 68.02% | 0.7829 |

> **Production Choice:** **Logistic Regression** was selected as the final production engine due to its superior generalization, high ROC-AUC ($0.9164$), and calibrated probability outputs required for clinical risk assessment.

---

## 🗂️ Project Directory Structure

```text
├── heart_disease_cleveland.csv    # Primary benchmark clinical dataset
├── heart_disease_combined.csv     # Extended multi-center clinical dataset
├── heart_disease.py         # Pipeline builder, cross-validation & artifact export
├── app.py                         # Streamlit interactive diagnostic dashboard
├── heart_disease_model.pkl        # Serialized production pipeline artifact
└── README.md                      # Project documentation

```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository & Set Up Virtual Environment

```bash
git clone [https://github.com/NajeebAhmed69/Heart_disease_Prediction_Model.git](https://github.com/NajeebAhmed69/Heart_disease_Prediction_Model.git)
cd heart-disease-diagnostic-pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Run Training & Cross-Validation Benchmark

```bash
python heart_disease.py

```

This executes the 5-fold cross-validation tournament, evaluates on the 20% holdout test set, and exports `heart_disease_model.pkl`.

### 3. Launch the Streamlit Diagnostic UI

```bash
streamlit run app.py

```

Access the application in your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `StratifiedKFold`, `LogisticRegression`, Ensembles)
* **Model Serialization:** Pickle
* **Deployment:** Streamlit

```

```