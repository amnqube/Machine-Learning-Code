# ==========================================================
# 📌 UNIVERSAL XGBOOST REGRESSION TEMPLATE (FULLY EXPLAINED)
# Purpose: Predict any continuous numeric value 
# Examples: Stock Price Prediction, Business Revenue Forecasting, Sales Prediction, etc.
# ==========================================================

# 1️⃣ Import all required libraries
# pandas → for data handling
# numpy → for numerical calculations
# sklearn.model_selection → to split data into training and testing sets
# sklearn.metrics → to evaluate regression model performance
# xgboost → the ML library we're using (XGBRegressor is for regression tasks)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# 2️⃣ Load your dataset
# Replace 'your_dataset.csv' with your dataset file name (CSV format assumed)
# You can also load from Excel (pd.read_excel) or directly from a DataFrame
df = pd.read_csv("your_dataset.csv")  # <-- Change file name if needed

# 3️⃣ Separate Features (X) and Target (y)
# Features (X) = all columns that are used to make the prediction
# Target (y) = the column we want to predict (must be numeric for regression)
TARGET = "target_column_name"  # <-- Change this to your dataset's target column name
X = df.drop(columns=[TARGET])  # Drops the target column, leaves only features
y = df[TARGET]                 # Stores target column separately

# 4️⃣ Split data into Training and Testing sets
# train_test_split → separates data so we can train the model on one set (train) 
# and evaluate on unseen data (test)
# test_size=0.2 → 20% for testing, 80% for training
# random_state=42 → ensures same split every run (reproducibility)
# shuffle=True → randomly mixes rows before splitting (good for most datasets, except time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 5️⃣ Initialize the XGBoost Regressor
# XGBRegressor is the regression version of XGBoost (used when target is continuous)
# Key hyperparameters explained:
model = XGBRegressor(
    n_estimators=500,         # Number of boosting rounds (trees). More = better learning but slower.
    learning_rate=0.05,       # Step size shrinkage to prevent overfitting. Lower = slower learning but better accuracy.
    max_depth=6,              # Maximum tree depth. Higher = more complex model, risk overfitting.
    min_child_weight=1,       # Minimum sum of instance weights in a child node. Higher = more conservative splits.
    gamma=0,                  # Minimum loss reduction for a split. Higher = more conservative.
    subsample=0.8,            # Fraction of training data used per tree. Less than 1.0 adds randomness → prevents overfitting.
    colsample_bytree=0.8,     # Fraction of features used per tree. Adds randomness to reduce overfitting.
    reg_alpha=0,              # L1 regularization (Lasso). Higher = more feature selection effect.
    reg_lambda=1,             # L2 regularization (Ridge). Higher = more penalty on large weights.
    random_state=42,          # Ensures reproducibility.
    n_jobs=-1,                # Use all CPU cores for faster training.
    eval_metric='rmse'        # Evaluation metric: RMSE (Root Mean Squared Error) for regression.
)

# 6️⃣ Train the model
# The model learns patterns from the training data
model.fit(X_train, y_train)

# 7️⃣ Make predictions on the test set
# Predicts target values for unseen test data
y_pred = model.predict(X_test)

# 8️⃣ Evaluate the model's performance
# RMSE → measures average prediction error size (lower = better)
# MAE → measures absolute prediction error (less sensitive to outliers than RMSE)
# R² Score → measures how well model explains variance (1 = perfect fit)
print("📊 Model Evaluation Metrics")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# 9️⃣ Predict for new data (optional)
# Make sure new_data has the SAME feature columns (order & names) as X
# Example:
# new_data = pd.DataFrame([[val1, val2, val3, ...]], columns=X.columns)
# prediction = model.predict(new_data)
# print("Predicted value:", prediction[0])