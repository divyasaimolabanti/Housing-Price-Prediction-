import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from flask import Flask, request, jsonify

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)
display(df.head())

# Step 2: Data Preprocessing
# Selecting relevant columns
cols = ["median_house_value", "median_income", "total_rooms", "housing_median_age", "ocean_proximity"]
df = df[cols]

# Handle missing values
df.dropna(inplace=True)

# Encode categorical variables
categorical_features = ["ocean_proximity"]
numerical_features = [col for col in df.columns if col not in ["ocean_proximity", "median_house_value"]]

# Visualizations
# Figure 1: Summary Statistics
print("Summary Statistics:")
print(df.describe())

# Figure 2: Histograms/Bar plots for Numeric Columns
df[numerical_features].hist(bins=20, figsize=(10, 6))
plt.suptitle("Histograms for Numerical Features")
plt.show()

# Figure 3: Correlation Matrix
corr_matrix = df[numerical_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# Step 3: Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Define features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4: Model Building with Grid Search
model = RandomForestRegressor()
param_grid = {"regressor__n_estimators": [50, 100, 200], "regressor__max_depth": [10, 20, 30]}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", model)
])

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Best Model Prediction
y_pred = grid_search.best_estimator_.predict(X_test)

# Step 5: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Evaluation Metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

# Model Evaluation Plot (Residuals)
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=(y_test - y_pred), lowess=True, color="g", line_kws={'color': 'red', 'lw': 1})
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Feature Importances (for Random Forest)
feature_importances = grid_search.best_estimator_.named_steps['regressor'].feature_importances_
features = numerical_features + list(grid_search.best_estimator_.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())
feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp_df)
plt.title("Feature Importances (Random Forest)")
plt.show()

# Save the model
joblib.dump(grid_search.best_estimator_, "house_price_model.pkl")

