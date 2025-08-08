# ==========================================================
# 📌 UNIVERSAL XGBOOST CLASSIFICATION TEMPLATE (FULLY EXPLAINED)
# Purpose: Predict a category/class label
# Examples: Fraud Detection, Customer Churn Prediction, Product Category Classification, etc.
# ==========================================================

# 1️⃣ Import all required libraries
# pandas → for data handling
# numpy → for numerical calculations
# sklearn.model_selection → to split data into training and testing sets
# sklearn.metrics → to evaluate classification model performance
# xgboost → the ML library we're using (XGBClassifier is for classification tasks)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# 2️⃣ Load your dataset
# Replace 'your_dataset.csv' with your dataset file name (CSV format assumed)
df = pd.read_csv("your_dataset.csv")  # <-- Change file name if needed

# 3️⃣ Separate Features (X) and Target (y)
# Features (X) = all columns that are used to make the prediction
# Target (y) = the column we want to predict (must be categorical for classification)
TARGET = "target_column_name"  # <-- Change this to your dataset's target column name
X = df.drop(columns=[TARGET])  # Drops the target column, leaves only features
y = df[TARGET]                 # Stores target column separately

# 4️⃣ Split data into Training and Testing sets
# test_size=0.2 → 20% for testing, 80% for training
# random_state=42 → ensures same split every run (reproducibility)
# shuffle=True → randomly mixes rows before splitting (good for most classification datasets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 5️⃣ Initialize the XGBoost Classifier
# XGBClassifier is the classification version of XGBoost
# Key hyperparameters explained:
model = XGBClassifier(
    n_estimators=500,         # Number of boosting rounds (trees). More trees = better learning but slower.
    learning_rate=0.05,       # Step size shrinkage to prevent overfitting. Lower values improve generalization.
    max_depth=6,              # Maximum tree depth. Higher depth captures more patterns but risks overfitting.
    min_child_weight=1,       # Minimum sum of instance weights in a child node. Controls overfitting.
    gamma=0,                  # Minimum loss reduction for a split. Higher values make the algorithm more conservative.
    subsample=0.8,            # Fraction of rows to use per tree. Lower values add randomness and prevent overfitting.
    colsample_bytree=0.8,     # Fraction of features per tree. Adds randomness to avoid overfitting.
    reg_alpha=0,              # L1 regularization (Lasso). Encourages sparsity in features.
    reg_lambda=1,             # L2 regularization (Ridge). Penalizes large weights.
    random_state=42,          # For reproducibility.
    n_jobs=-1,                # Use all CPU cores for faster training.
    eval_metric='logloss',    # Evaluation metric: 'logloss' for classification.
    use_label_encoder=False   # Avoids warning messages for label encoding.
)

# 6️⃣ Train the model
# The model learns from the training data
model.fit(X_train, y_train)

# 7️⃣ Make predictions on the test set
y_pred = model.predict(X_test)

# 8️⃣ Evaluate the model's performance
# Accuracy → percentage of correct predictions
# Classification Report → precision, recall, f1-score for each class
# Confusion Matrix → breakdown of correct/incorrect predictions
print("📊 Model Evaluation Metrics")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9️⃣ Predict for new data (optional)
# Ensure new_data has the SAME feature columns as X
# Example:
# new_data = pd.DataFrame([[val1, val2, val3, ...]], columns=X.columns)
# prediction = model.predict(new_data)
# print("Predicted Class:", prediction[0])