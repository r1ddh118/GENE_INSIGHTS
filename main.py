import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency

# Load the datasets
alzheimers_data = pd.read_csv('C:\\Users\\Victus\\OneDrive\\Desktop\\alzheimers_disease_data.csv')
genome_data = pd.read_csv('C:\\Users\\Victus\\OneDrive\\Desktop\\1000-genomes-phase-3_reports_1000_Genomes_phase_3_sample_results.csv')

# 1. Data Cleaning
# Check for missing values and remove rows with missing data
print("Missing values in Alzheimer's data:\n", alzheimers_data.isnull().sum())
print("Missing values in Genome data:\n", genome_data.isnull().sum())

alzheimers_data.dropna(inplace=True)
genome_data.dropna(inplace=True)

# Validate data labels
assert alzheimers_data['Diagnosis'].isin([0, 1]).all(), "Diagnosis column contains invalid labels"

# 2. Feature Calculation
# Example: Platelet-to-Lymphocyte Ratio (replace 'Platelets' and 'Lymphocytes' with actual column names if available)
if 'Platelets' in alzheimers_data.columns and 'Lymphocytes' in alzheimers_data.columns:
    alzheimers_data['PLR'] = alzheimers_data['Platelets'] / alzheimers_data['Lymphocytes']

# Normalize selected features
features = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP']
scaler = StandardScaler()
alzheimers_data[features] = scaler.fit_transform(alzheimers_data[features])

# 3. Exploratory Data Analysis
# Histograms of each parameter
for feature in features:
    plt.figure()
    sns.histplot(alzheimers_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Box plot for each feature to identify outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=alzheimers_data[features])
plt.title('Box Plot of Features')
plt.show()

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(alzheimers_data[features + ['Diagnosis']].corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

# Adjusted Scatter Plots for Diagnosis vs Features
feature_pairs = [('DiastolicBP', 'Diagnosis'), ('SystolicBP', 'Diagnosis'), ('BMI', 'Diagnosis'), ('Age', 'Diagnosis')]
for feature, target in feature_pairs:
    plt.figure()
    sns.regplot(x=alzheimers_data[feature], y=alzheimers_data[target], logistic=True, scatter_kws={"s": 10})
    plt.title(f'{feature} vs {target}')
    plt.show()

# 4. Identifying Homozygous Alternate Variants
# Identify rows in genome data with homozygous alternate variants
if 'heterozygous_variant_count' in genome_data.columns:
    homozygous_variants = genome_data[genome_data['heterozygous_variant_count'] == 0]

# 5. Identifying Biomarkers and Therapeutic Targets
# Association Test: Chi-square test to identify significant association with Diagnosis
significant_variants = []
for column in genome_data.select_dtypes(include=['int', 'float']).columns:
    contingency_table = pd.crosstab(alzheimers_data['Diagnosis'], genome_data[column])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    if p < 0.05:  # significant if p < 0.05
        significant_variants.append(column)

print("Significant Variants associated with Alzheimer's Disease:", significant_variants)

# 6. Model Training and Evaluation
# Extract features and target for model
X = alzheimers_data[features]
y = alzheimers_data['Diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predict probabilities for each individual
alzheimers_data['Alzheimers_Probability'] = logistic_model.predict_proba(X)[:, 1]
alzheimers_data['Alzheimers_Probability_Percentage'] = alzheimers_data['Alzheimers_Probability'] * 100

# Display each individual's Alzheimer’s Probability
print(alzheimers_data[['PatientID', 'Alzheimers_Probability_Percentage']])

# Predict on the test set and evaluate accuracy
y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Additional Exploratory Visualizations
# Distribution of Alzheimer's Probability
plt.figure()
sns.histplot(alzheimers_data['Alzheimers_Probability_Percentage'], kde=True, color='purple')
plt.title("Distribution of Alzheimer's Probability Percentage")
plt.xlabel("Alzheimer's Probability (%)")
plt.show()

# Violin plot for Probability by Diagnosis
plt.figure()
sns.violinplot(x=alzheimers_data['Diagnosis'], y=alzheimers_data['Alzheimers_Probability_Percentage'])
plt.title("Alzheimer's Probability by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Alzheimer's Probability (%)")
plt.show()
