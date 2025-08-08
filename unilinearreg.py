# ==========================================================
# 📌 UNIVERSAL LINEAR REGRESSION TEMPLATE (FULLY EXPLAINED)
# Purpose: Predict any continuous numeric value using a simple linear relationship
# Examples: Predicting house prices, sales amounts, salaries, etc.
# ==========================================================

# 1️⃣ Import required libraries
# pandas → for data handling
# numpy → for numerical operations
# sklearn.model_selection → to split data into training/testing
# sklearn.linear_model → for LinearRegression model
# sklearn.metrics → to evaluate regression performance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 2️⃣ Load your dataset
# Replace 'your_dataset.csv' with your file name (can be Excel, SQL, etc.)
df = pd.read_csv("your_dataset.csv")  # <-- Change as needed

# 3️⃣ Separate Features (X) and Target (y)
# X = independent variables (features)
# y = dependent variable (the numeric value we want to predict)
TARGET = "target_column_name"  # <-- Change to your target column
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 4️⃣ Split into Training and Testing sets
# test_size=0.2 → 20% data for testing
# random_state → ensures reproducibility
# shuffle=True → good for most datasets (except time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 5️⃣ Initialize the Linear Regression Model
# Hyperparameters explained:
model = LinearRegression(
    fit_intercept=True,   # Whether to calculate the intercept (b0). Set False if data already centered.
    copy_X=True,          # Whether to copy input data. If False, may overwrite X (memory efficient).
    n_jobs=-1,            # Number of CPU cores to use (-1 = all cores).
    positive=False        # If True, forces coefficients to be positive (useful in some business constraints).
)

# 6️⃣ Train the model (fit on training data)
model.fit(X_train, y_train)

# 7️⃣ Make predictions
y_pred = model.predict(X_test)

# 8️⃣ Evaluate performance
print("📊 Model Evaluation Metrics")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")  # Lower = better
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")           # Less sensitive to outliers
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")                 # 1.0 = perfect prediction

# 9️⃣ Check model coefficients
# Coefficients tell how much each feature impacts the prediction
print("\n📌 Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

# Intercept (b0 in equation y = b0 + b1*x1 + b2*x2 + ...)
print(f"\nIntercept (b0): {model.intercept_:.4f}")

# 🔟 Predict for new data (optional)
# Ensure new_data has the SAME feature columns and order as X
# Example:
# new_data = pd.DataFrame([[val1, val2, val3, ...]], columns=X.columns)
# prediction = model.predict(new_data)
# print("Predicted value:", prediction[0])