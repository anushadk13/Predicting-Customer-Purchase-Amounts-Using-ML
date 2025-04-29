# Predicting-Customer-Purchase-Amounts-Using-ML

Project: Predicting Customer Purchase Amounts Using ML

1. Project Setup

Install required libraries:
pip install pandas numpy scikit-learn matplotlib seaborn
Libraries we'll use:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score

2. Data Collection

Get a dataset. Some options:
Kaggle ➔ "Black Friday Dataset" (commonly used for purchase prediction).
Or simulate your own synthetic data.
Example to load data:
df = pd.read_csv('train.csv')

3. Data Exploration (EDA)

Understand the data:
print(df.head())
print(df.info())
print(df.describe())
Check missing values:
print(df.isnull().sum())
Visualize distributions:
sns.histplot(df['Purchase'])
plt.show()
Correlation matrix:
sns.heatmap(df.corr(), annot=True)
plt.show()

4. Data Preprocessing

Fill missing values:
df.fillna(df.median(), inplace=True)
Encode categorical variables:
df = pd.get_dummies(df, drop_first=True)
Separate features and target:
X = df.drop('Purchase', axis=1)
y = df['Purchase']
Train-Test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
(Optional) Scale features:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

5. Base Models

Create several regression models:
base_models = [
    ('lr', LinearRegression()),
    ('ridge', Ridge(alpha=1.0)),
    ('lasso', Lasso(alpha=0.1)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42))
]

6. Cross Validation

Evaluate each model with K-Fold Cross Validation:
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in base_models:
    scores = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=kf)
    print(f"{name}: RMSE: {-scores.mean():.4f}")
    
7. Stacking

Combine the base models into a Stacking Regressor:
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=42)
)
Train the Stacking model:
stacking_model.fit(X_train, y_train)

8. Model Evaluation

Predict and evaluate:
y_pred = stacking_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2 Score: {r2:.4f}")
Plot predictions:
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Purchase Amounts')
plt.show()

Save the model:
import joblib
joblib.dump(stacking_model, 'stacked_purchase_model.pkl')
Later load it:
model = joblib.load('stacked_purchase_model.pkl')
