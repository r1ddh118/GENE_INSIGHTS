import numpy as np
import pandas as pd

# Load the dataset (replace with your actual dataset file path)
df = pd.read_csv('/content/alzheimers_disease_data.csv')

# Define the possible APOE genotypes and their probabilities in the population
apoe_genotypes = ['e2/e2', 'e2/e3', 'e2/e4', 'e3/e3', 'e3/e4', 'e4/e4']
genotype_probabilities = [0.01, 0.13, 0.03, 0.60, 0.20, 0.03]  # Example probabilities

# Add a new column 'APOE_genotype' with random values based on these probabilities
np.random.seed(42)  # For reproducibility
df['APOE_genotype'] = np.random.choice(apoe_genotypes, size=len(df), p=genotype_probabilities)

# Display the first few rows to check the new column
print(df.head())

# You can now proceed to include the 'APOE_genotype' column in your analysis

apoe_risk_mapping = {
    'e2/e2': 0,  # Low risk
    'e2/e3': 0,  # Low risk
    'e3/e3': 1,  # Normal risk
    'e2/e4': 1,  # Intermediate risk
    'e3/e4': 1,  # Intermediate risk
    'e4/e4': 2   # High risk
}

# Apply the mapping to the 'APOE_genotype' column
df['APOE_risk_score'] = df['APOE_genotype'].map(apoe_risk_mapping)

# Check if the transformation worked
print(df[['APOE_genotype', 'APOE_risk_score']].head())

df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)

df

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Handle categorical variables (like Gender, Ethnicity, etc.)
df = pd.get_dummies(df, drop_first=True)

# Define feature variables (X) and the target (y)
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Check feature importance (coefficients)
importance = model.coef_[0]
for i, v in enumerate(importance):
    print(f'Feature: {X.columns[i]}, Importance: {v}')

# prompt: give the code after the previous cell

from sklearn.metrics import classification_report, confusion_matrix

# Generate classification report
print(classification_report(y_test, y_pred))

# Generate confusion matrix
print(confusion_matrix(y_test, y_pred))

# Analyze the results, look for areas of improvement (e.g., handling class imbalance, exploring different models).
# You might want to try other models like SVM, Random Forest, or XGBoost to see if they perform better.

# You can also visualize the data and model predictions for better understanding.

