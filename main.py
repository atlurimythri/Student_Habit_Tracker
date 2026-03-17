import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
print("========== STEP 1: Loading Processed Data ==========")
df = pd.read_csv("processed_student_data.csv")
print(df.head())
print("\n========== STEP 2: Defining Features (X) and Target (y) ==========")
X = df.drop(columns=["effectiveness_score", "student_id"])
y = df["effectiveness_score"]
print("\nFeatures (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())
print("\n========== STEP 3: Splitting Data into Train & Test ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("\n========== STEP 4: Training Linear Regression Model ==========")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed.")
print("\n========== STEP 5: Making Predictions ==========")
y_pred = model.predict(X_test)
print("Predicted values:")
print(y_pred[:5])  
print("\n========== STEP 6: Evaluating Model ==========")
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("R2 Score:", r2)
print("Mean Squared Error:", mse)
print("\n========== STEP 7: Correlation Heatmap ==========")
corr = df.corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
print("\n========== STEP 8: Comparing Actual vs Predicted ==========")
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print(comparison.head(10))
print("\n========== STEP 9: Feature Importance ==========")
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(feature_importance.sort_values(by="Coefficient", ascending=False))
print("\n========== STEP 10: Feature Importance ==========")
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(feature_importance.sort_values(by="Coefficient", ascending=False))
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved successfully!")