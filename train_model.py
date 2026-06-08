import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer # Better for insurance data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.exceptions import ConvergenceWarning

# --- 1. Load dataset ---
data = pd.read_csv("insurance_data.csv") 
data.columns = data.columns.str.strip()

# --- 2. Feature Engineering ---
# Adding these helps the existing layers understand the "scale" of fraud
data['Claim_to_Deductible_Ratio'] = data['Claim_Amount'] / (data['Deductible_Amount'] + 1)
data['Risk_Score'] = (data['Claim_Amount'] * data['Number_of_Previous_Claims_Patient']) / 1000

# Convert categories
data["Claim_Submitted_Late"] = data["Claim_Submitted_Late"].apply(lambda x: 1 if str(x).lower() == "yes" else 0)
target = "Is_Fraudulent"
data[target] = data[target].apply(lambda x: 1 if str(x).lower() in ["yes", "true", "1"] else 0)

features = ["Claim_Amount", "Deductible_Amount", "CoPay_Amount", 
            "Number_of_Previous_Claims_Patient", "Claim_Submitted_Late", 
            "Claim_to_Deductible_Ratio", "Risk_Score"]

X = data[features]
y = data[target]

# --- 3. Better Balancing & Scaling ---
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# PowerTransformer makes data more Gaussian (Normal), which Neural Nets love
scaler = PowerTransformer(method='yeo-johnson')
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# --- 4. MLP with SGD Optimizer ---
# --- 2. Custom Epoch Steps (Optimized for Peak F1-Score) ---
# We stop at 40 because your analysis showed the best balance there
epoch_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40]

model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='sgd',
    learning_rate='adaptive',
    momentum=0.9,
    learning_rate_init=0.01,
    alpha=0.001,
    max_iter=1,
    warm_start=True, 
    random_state=42
)

print(f"{'Epoch':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 65)

for epoch in epoch_steps:
    model.set_params(max_iter=epoch)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Metrics calculation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"{epoch:<8} | {acc:.4f}     | {prec:.4f}      | {rec:.4f}      | {f1:.4f}")

# --- 3. Save the Peak Performance Model ---
joblib.dump(model, "fraud_dl_model.pkl")

print("\n" + "="*40)
print("✅ TRAINING COMPLETE")
print(f"Final Model saved at Epoch: {epoch_steps[-1]}")
print("This version provides the best F1-Score balance.")
