# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Step 1: Read data
df = pd.read_csv("xyz.csv")
print(df)

# Step 2: Split X and y
X = df[['X']]
y = df['y']

# Step 3: Train-test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Polynomial transformation (to make curve)
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Step 5: Create feature names
feature_names = poly.get_feature_names_out(['X'])

# Step 6: Train Linear, Ridge, and Lasso models
model1 = LinearRegression()
model2 = Ridge(alpha=10)
model3 = Lasso(alpha=0.1)

model1.fit(X_train_poly, y_train)
model2.fit(X_train_poly, y_train)
model3.fit(X_train_poly, y_train)

# Step 7: Predictions
y_pred1 = model1.predict(X_test_poly)
y_pred2 = model2.predict(X_test_poly)
y_pred3 = model3.predict(X_test_poly)

# Step 8: R² scores
print("\n--- R² SCORES ---")
print("Linear Regression:", r2_score(y_test, y_pred1))
print("Ridge Regression:", r2_score(y_test, y_pred2))
print("Lasso Regression:", r2_score(y_test, y_pred3))

# Step 9: Plot comparison
plt.scatter(X, y, color='black', label='Actual Data')
plt.plot(np.sort(X_train, axis=0),
         model1.predict(poly.transform(np.sort(X_train, axis=0))),
         color='red', label='Linear (Overfit)')
plt.plot(np.sort(X_train, axis=0),
         model2.predict(poly.transform(np.sort(X_train, axis=0))),
         color='blue', label='Ridge (Regularized)')
plt.plot(np.sort(X_train, axis=0),
         model3.predict(poly.transform(np.sort(X_train, axis=0))),
         color='green', label='Lasso (Feature Selection)')
plt.title("Overfitting vs Ridge vs Lasso (Polynomial Features)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Step 10: Identify bad (irrelevant) features where Lasso coefficients = 0
bad_features = np.where(model3.coef_ == 0)[0]
print("\nIndexes of irrelevant features:", bad_features)
print("Names of irrelevant features:", list(np.array(feature_names)[bad_features]))

# Step 11: Remove bad features from polynomial data
X_train_poly_df = pd.DataFrame(X_train_poly, columns=feature_names)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=feature_names)

X_train_filtered = X_train_poly_df.drop(X_train_poly_df.columns[bad_features], axis=1)
X_test_filtered = X_test_poly_df.drop(X_test_poly_df.columns[bad_features], axis=1)

print("\nShape before filtering:", X_train_poly_df.shape)
print("Shape after filtering:", X_train_filtered.shape)