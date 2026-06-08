from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from functools import wraps
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = "insurance_corporate_secret_key_2024"

# -----------------------------
# Load Model
# -----------------------------
try:
    model = joblib.load("fraud_dl_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    model, scaler = None, None

# -----------------------------
# Load Dataset
# -----------------------------
try:
    historical_data = pd.read_csv("processed_data.csv")
    historical_data.columns = historical_data.columns.str.strip()
except:
    historical_data = None

# -----------------------------
# History File
# -----------------------------
HISTORY_FILE = "prediction_history.csv"
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["timestamp","claim_amount","risk_score","result"]).to_csv(HISTORY_FILE,index=False)

# -----------------------------
# Employee DB
# -----------------------------
EMPLOYEE_DB = {
    "admin_112": {"pw": "secure123", "role": "Admin"},
    "investigator_118": {"pw": "detective2024", "role": "Investigator"},
    "adjuster_114": {"pw": "claims1234", "role": "Adjuster"}
}

# -----------------------------
# Login Required
# -----------------------------
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap

# -----------------------------
# Graph Helper
# -----------------------------
def make_graph(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return graph_url

def fraud_graph():
    if historical_data is None:
        return None

    fig = plt.figure()
    data = historical_data["Is_Fraudulent"].map({0: "Non-Fraud", 1: "Fraud"})
    data.value_counts().plot(kind="bar")
    plt.title("Fraud vs Non-Fraud")
    plt.xticks(rotation=0)

    return make_graph(fig)

def claim_amount_graph():
    if historical_data is None:
        return None

    fig = plt.figure()
    historical_data["Claim_Amount"].plot(kind='hist', bins=20)
    plt.title("Claim Amount Distribution")

    return make_graph(fig)

# -----------------------------
# Routes
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")

        if u in EMPLOYEE_DB and EMPLOYEE_DB[u]["pw"] == p:
            session["user"] = u
            return redirect(url_for("home"))
        else:
            error = "Invalid Credentials"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def home():
    return render_template("index.html")

# -----------------------------
# PREDICT (FINAL FIXED)
# -----------------------------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        return "Model not loaded"

    try:
        amt = float(request.form.get("claim_amount", 0))
        ded = float(request.form.get("deductible", 0))
        copay = float(request.form.get("copay", 0))
        prev = float(request.form.get("previous_claims", 0))
        late = 1 if request.form.get("late") == "Yes" else 0

        # -----------------------------
        # Feature Engineering
        # -----------------------------
        ratio = amt / (ded + 1)
        risk_sc = (amt * prev) / 1000

        try:
            # New model (7 features)
            features = np.array([[amt, ded, copay, prev, late, ratio, risk_sc]])
            scaled = scaler.transform(features)
        except:
            # Old model fallback (5 features)
            features = np.array([[amt, ded, copay, prev, late]])
            scaled = scaler.transform(features)

        print("Input Shape:", features.shape)

        # -----------------------------
        # Prediction
        # -----------------------------
        try:
            prob = model.predict_proba(scaled)[0][1]
        except:
            prob = float(model.predict(scaled)[0])

        print("Fraud Probability:", prob)

        risk_score = int(prob * 100)

        # -----------------------------
        # IMPROVED THRESHOLDS
        # -----------------------------
        if prob >= 0.75:
            txt = "🚨 High Risk (Fraud Likely)"
            color = "#e74c3c"
            result = "High"

        elif prob >= 0.45:
            txt = "⚠️ Moderate Risk"
            color = "#f39c12"
            result = "Moderate"

        else:
            txt = "✅ Low Risk (Genuine Claim)"
            color = "#2ecc71"
            result = "Low"

        # -----------------------------
        # RULE-BASED CORRECTIONS
        # -----------------------------
        if amt < 50000 and prev <= 2 and late == 0:
            txt = "✅ Low Risk (Consistent Genuine Pattern)"
            color = "#2ecc71"
            result = "Low"
            risk_score = min(risk_score, 20)

        if amt > 150000 and prev > 5:
            txt = "🚨 High Risk (Suspicious Pattern)"
            color = "#e74c3c"
            result = "High"
            risk_score = max(risk_score, 85)

        # -----------------------------
        # Explanations
        # -----------------------------
        explanations = []
        if amt > 75000:
            explanations.append("The claim exhibits a significantly elevated claim amount")
        if prev > 5:
            explanations.append("The claim exhibits a pattern of frequent claim activity")
        if late:
            explanations.append("Submission beyond the permissible reporting timeframe")

        # -----------------------------
        # Save History
        # -----------------------------
        pd.DataFrame([{
            "timestamp": datetime.now(),
            "claim_amount": amt,
            "risk_score": risk_score,
            "result": result
        }]).to_csv(HISTORY_FILE, mode='a', header=False, index=False)

        # -----------------------------
        # Comparison
        # -----------------------------
        percentile = 0
        if historical_data is not None:
            percentile = int((historical_data["Claim_Amount"] < amt).mean() * 100)

        comparison = f"Your claim is higher than {percentile}% of past claims."

        return render_template(
            "index.html",
            risk=risk_score,
            prediction_text=txt,
            color=color,
            explanation=explanations,
            comparison=comparison
        )

    except Exception as e:
        print("ERROR:", e)
        return redirect(url_for("home"))

# -----------------------------
# DASHBOARD
# -----------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        g1=fraud_graph(),
        g2=claim_amount_graph()
    )

# -----------------------------
# HISTORY
# -----------------------------
@app.route("/history")
@login_required
def history():
    try:
        data = pd.read_csv(HISTORY_FILE)
    except:
        data = pd.DataFrame()

    return render_template("history.html", data=data)

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)