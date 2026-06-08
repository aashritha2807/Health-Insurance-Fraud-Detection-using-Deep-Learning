# 🏥 Health Insurance Fraud Detection Using Deep Learning

A web-based application that uses a trained MLP (Multi-Layer Perceptron) deep learning model to detect potentially fraudulent health insurance claims. The system provides real-time risk scoring, explainable AI (XAI) reasoning, and a historical audit trail — all behind a secure employee login portal.

---

## 📁 Project Structure

```
project/
│
├── app.py                    # Flask backend — routes, prediction logic, auth
├── train_model.py            # Model training script (MLP with SMOTE + PowerTransformer)
│
├── templates/
│   ├── login.html            # Employee login page
│   ├── index.html            # Main prediction interface
│   ├── dashboard.html        # Analytics dashboard (charts)
│   └── history.html          # Prediction audit history
│
├── models/
│   ├── fraud_dl_model.pkl    # Trained MLP classifier
│   ├── fraud_model.pkl       # Alternate/older model
│   ├── fraud_model_balanced.pkl
│   ├── fraud_dnn_model.keras # Keras DNN model
│   └── scaler.pkl            # PowerTransformer scaler
│
├── data/
│   ├── insurance_data.csv    # Raw training dataset
│   ├── processed_data.csv    # Preprocessed dataset used for comparison
│   └── prediction_history.csv # Logged predictions (appended on each run)
│
└── log.txt                   # Debug/output log
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install flask numpy pandas scikit-learn imbalanced-learn matplotlib joblib
```

For the Keras model (optional):
```bash
pip install tensorflow
```

### 2. Train the Model (optional — pre-trained model included)

```bash
python train_model.py
```

This will:
- Load `insurance_data.csv`
- Apply SMOTE oversampling to handle class imbalance
- Scale features with `PowerTransformer`
- Train an MLP across epoch checkpoints (1 → 40)
- Save `fraud_dl_model.pkl` and `scaler.pkl`

### 3. Run the Application

```bash
python app.py
```

Then open your browser to: `http://127.0.0.1:5000`

---

## 🔐 Login Credentials

| Employee ID       | Password       | Role          |
|-------------------|----------------|---------------|
| admin_112         | secure123      | Admin         |
| investigator_118  | detective2024  | Investigator  |
| adjuster_114      | claims1234     | Adjuster      |

> ⚠️ These are demo credentials. Replace with a secure auth system before production deployment.

---

## 🧠 Model Details

| Parameter              | Value                          |
|------------------------|--------------------------------|
| Algorithm              | MLPClassifier (scikit-learn)   |
| Architecture           | 64 → 32 → 16 (ReLU)           |
| Optimizer              | SGD with momentum (0.9)        |
| Learning Rate          | 0.01 (adaptive)                |
| Regularization         | L2 alpha = 0.001               |
| Class Balancing        | SMOTE                          |
| Feature Scaling        | PowerTransformer (Yeo-Johnson) |
| Best Epoch             | 40                             |

### Input Features

| Feature                        | Description                          |
|--------------------------------|--------------------------------------|
| Claim Amount                   | Total value of the insurance claim   |
| Deductible Amount              | Patient's deductible                 |
| CoPay Amount                   | Patient's copay                      |
| Number of Previous Claims      | Historical claim count for patient   |
| Claim Submitted Late           | Whether claim was filed past deadline|
| Claim-to-Deductible Ratio      | Engineered: `amount / (deductible+1)`|
| Risk Score                     | Engineered: `(amount × prev_claims) / 1000` |

---

## 📊 Risk Thresholds

| Fraud Probability | Risk Level | Label                          |
|-------------------|------------|--------------------------------|
| ≥ 75%             | 🔴 High    | Fraud Likely                   |
| 45% – 74%         | 🟡 Moderate| Requires Review                |
| < 45%             | 🟢 Low     | Genuine Claim                  |

### Rule-Based Overrides

- If `claim_amount < 50,000` AND `previous_claims ≤ 2` AND `not late` → forced **Low Risk** (score capped at 20)
- If `claim_amount > 150,000` AND `previous_claims > 5` → forced **High Risk** (score floored at 85)

---

## 🔍 Explainable AI (XAI)

The system generates plain-language explanations for each prediction:

- *"The claim exhibits a significantly elevated claim amount"* — triggered when amount > ₹75,000
- *"The claim exhibits a pattern of frequent claim activity"* — triggered when previous claims > 5
- *"Submission beyond the permissible reporting timeframe"* — triggered when filed late

---

## 📈 Features

- **Fraud Risk Score** — Visual circular gauge (0–100%) with color coding
- **Explainable Statements** — Human-readable reasons for the model's decision
- **Claim Comparison** — Tells you what percentile the claim falls in vs. historical data
- **Dashboard** — Bar chart of fraud distribution + histogram of claim amounts
- **Prediction History** — Full audit log of all past predictions with timestamps

---

## ⚠️ Disclaimer

This system provides AI-based fraud risk assessments only. Results are probabilistic predictions and **must not** be used as final decisions. All flagged claims require review by authorized personnel before any action is taken.

---

## 🛠️ Future Improvements

- Replace hardcoded credentials with a proper database + hashed passwords
- Add role-based access control (Admin vs. Investigator views)
- Integrate SHAP values for deeper explainability
- Export prediction history as PDF/Excel reports
- Add REST API endpoints for integration with external claim management systems
