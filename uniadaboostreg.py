# ==========================================================
# 📌 UNIVERSAL ADABOOST REGRESSION TEMPLATE (FULLY EXPLAINED)
# Purpose: Predict continuous numeric values
# Examples: Stock Price Prediction, Sales Forecasting, Temperature Prediction
# ==========================================================

# 1️⃣ Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor  # Default base learner for AdaBoost Regression

# 2️⃣ Load dataset
df = pd.read_csv("your_dataset.csv")  # Change file name if needed

# 3️⃣ Define target
TARGET = "target_column_name"  # <-- Change to your dataset's target column

# 4️⃣ Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 5️⃣ Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 6️⃣ Initialize AdaBoost Regressor
model = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(
        max_depth=4,          # Controls complexity of each weak learner
        min_samples_split=2,  # Minimum samples to split a node
        min_samples_leaf=1    # Minimum samples at a leaf
    ),
    n_estimators=100,         # Number of boosting stages
    learning_rate=0.1,        # Smaller = slower learning but better generalization
    loss='linear',            # Options: 'linear', 'square', 'exponential'
    random_state=42
)

# 7️⃣ Train the model
model.fit(X_train, y_train)

# 8️⃣ Predict on test set
y_pred = model.predict(X_test)

# 9️⃣ Evaluate performance
print("📊 Model Evaluation Metrics")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# 🔟 Predict for new data
# new_data = pd.DataFrame([[val1, val2, val3, ...]], columns=X.columns)
# prediction = model.predict(new_data)
# print("Predicted Value:", prediction[0])