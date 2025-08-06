# Telco Customer Churn Prediction using AdaBoost with interactive user input
# Compatible with scikit-learn >= 1.2 (uses 'estimator' instead of deprecated 'base_estimator')

# === 1. Import Required Libraries ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import joblib

# === 2. Load Dataset ===

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# === 3. Preprocessing ===

# Drop 'customerID' column
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Encode target column
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode binary categorical columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# One-hot encode remaining categorical features
df = pd.get_dummies(df, drop_first=True)

# === 4. Train-Test Split ===

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 5. Feature Scaling ===

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. Model Building (Updated for scikit-learn >= 1.2) ===

model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# === 7. Evaluation ===

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# === 8. Confusion Matrix ===

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# === 9. ROC Curve ===

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# === 10. Feature Importances ===

importances = model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x='Importance', y='Feature')
plt.title("Top 15 Important Features")
plt.tight_layout()
plt.show()

# === 11. Save Model & Scaler ===

joblib.dump(model, 'adaboost_telco_model.pkl')
joblib.dump(scaler, 'scaler_telco.pkl')

# === 12. Predict New Data (Line-by-line input) ===

print("\n--- Input New Customer Data for Prediction ---")
print("Instructions: If you want to skip a feature, type 'noinput' (will use 0).\n")

# Load again for safety
model = joblib.load('adaboost_telco_model.pkl')
scaler = joblib.load('scaler_telco.pkl')

# Prepare user input
user_input = {}
for feature in X.columns:
    if "Yes" in feature or "No" in feature:
        hint = "(yes=1 / no=0)"
    elif feature in ['MonthlyCharges', 'TotalCharges', 'tenure']:
        hint = "(numeric)"
    elif feature in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        hint = "(binary: 0 or 1)"
    else:
        hint = "(0/1 or numeric depending on encoded column)"

    val = input(f"{feature} {hint}: ").strip().lower()

    if val == "noinput":
        user_input[feature] = 0
    else:
        try:
            user_input[feature] = float(val)
        except:
            print(f"Invalid input for {feature}, defaulting to 0.")
            user_input[feature] = 0

# Convert to DataFrame and scale
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# Predict
pred = model.predict(input_scaled)[0]
pred_prob = model.predict_proba(input_scaled)[0][1]

print("\n--- Prediction Result ---")
print("🔴 Likely to Churn" if pred == 1 else "🟢 Not Likely to Churn")
print(f"Probability of Churn: {pred_prob:.2f}")
