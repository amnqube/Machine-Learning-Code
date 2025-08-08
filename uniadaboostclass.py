# ==========================================================
# 📌 UNIVERSAL ADABOOST CLASSIFICATION TEMPLATE (FULLY EXPLAINED)
# Purpose: Predict any categorical (class) target variable
# Examples: Spam Detection, Customer Churn, Disease Prediction, etc.
# ==========================================================

# 1️⃣ Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier  # Default base learner for AdaBoost

# 2️⃣ Load your dataset
df = pd.read_csv("your_dataset.csv")  # Change file name as needed

# 3️⃣ Define target column
TARGET = "target_column_name"  # <-- Change to your dataset's target column

# 4️⃣ Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 5️⃣ Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,         # 20% test data
    random_state=42,       # reproducibility
    shuffle=True           # shuffle for randomness (avoid for time series)
)

# 6️⃣ Initialize AdaBoost Classifier
model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(
        max_depth=3,         # Depth of individual trees (shallow = less overfitting)
        min_samples_split=2, # Minimum samples to split a node
        min_samples_leaf=1   # Minimum samples at a leaf node
    ),
    n_estimators=50,          # Number of weak learners (trees)
    learning_rate=1.0,        # Shrinks the contribution of each tree
    algorithm='SAMME.R',      # 'SAMME.R' = uses probabilities (better, faster), 'SAMME' = discrete boosting
    random_state=42
)

# 7️⃣ Train the model
model.fit(X_train, y_train)

# 8️⃣ Predict on test set
y_pred = model.predict(X_test)

# 9️⃣ Evaluate performance
print("📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 🔟 Predict for new data
# new_data = pd.DataFrame([[val1, val2, val3, ...]], columns=X.columns)
# prediction = model.predict(new_data)
# print("Predicted Class:", prediction[0])