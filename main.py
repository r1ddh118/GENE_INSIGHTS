import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency

# Load datasets
alzheimers_data = pd.read_csv('/home/r1ddh1/2nd_year/dpel_project_things/Gene Insights/GENE_INSIGHTS/Datasets/alzheimers_disease_data.csv')
genome_data = pd.read_csv('/home/r1ddh1/2nd_year/dpel_project_things/Gene Insights/GENE_INSIGHTS/Datasets/1000-genomes-phase-3_reports_1000_Genomes_phase_3_sample_results.csv')

# Data cleaning
alzheimers_data.dropna(inplace=True)
genome_data.dropna(inplace=True)

# Validate Diagnosis column
assert alzheimers_data['Diagnosis'].isin([0, 1]).all(), "Diagnosis column contains invalid labels"

# Feature Calculation: Platelet-to-Lymphocyte Ratio if columns exist
if 'Platelets' in alzheimers_data.columns and 'Lymphocytes' in alzheimers_data.columns:
    alzheimers_data['PLR'] = alzheimers_data['Platelets'] / alzheimers_data['Lymphocytes']

# Feature normalization
features = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP']
scaler = StandardScaler()
alzheimers_data[features] = scaler.fit_transform(alzheimers_data[features])

# Train a Logistic Regression Model for Probability Prediction
X = alzheimers_data[features]
y = alzheimers_data['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Add probability predictions to dataset
alzheimers_data['Alzheimers_Probability'] = logistic_model.predict_proba(X)[:, 1]
alzheimers_data['Alzheimers_Probability_Percentage'] = alzheimers_data['Alzheimers_Probability'] * 100

# Set up Streamlit dashboard
st.title("Alzheimer's Probability Diagnosis Dashboard")
st.write("Interactive dashboard for Alzheimer's disease probability and other clinical insights.")

# Patient-specific data
selected_patient = st.selectbox("Select Patient ID", alzheimers_data['PatientID'].unique())
patient_data = alzheimers_data[alzheimers_data['PatientID'] == selected_patient]
probability = patient_data['Alzheimers_Probability_Percentage'].values[0]

# Display Gauge for Selected Patient's Probability
st.subheader("Alzheimer's Probability Gauge")
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability,
    title={'text': "Alzheimer's Probability (%)"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "purple"}}))
st.plotly_chart(fig)

# Patient-specific KDE plot
st.subheader("Patient-Specific KDE Plot for Alzheimer's Probability by Diagnosis")
fig, ax = plt.subplots()
sns.kdeplot(alzheimers_data['Alzheimers_Probability_Percentage'], label="All Patients", color='purple')
sns.kdeplot(patient_data['Alzheimers_Probability_Percentage'], label="Selected Patient", color='red')
plt.title("KDE Plot of Alzheimer's Probability")
st.pyplot(fig)

# Patient-specific Scatter Plot and overall Heatmap
st.subheader("Feature Correlation Heatmap with Patient-Specific Scatter Plot")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(alzheimers_data[features + ['Alzheimers_Probability']].corr(), annot=True, cmap="coolwarm")
ax.scatter([patient_data['Age'].values[0]], [patient_data['Alzheimers_Probability'].values[0]], color="red", label="Selected Patient")
plt.legend()
st.pyplot(fig)

# Overall Probability Distribution for All Patients
st.subheader("Distribution of Alzheimer's Probability for All Patients")
fig, ax = plt.subplots()
sns.histplot(alzheimers_data['Alzheimers_Probability_Percentage'], kde=True, color='purple')
plt.title("Distribution of Alzheimer's Probability")
st.pyplot(fig)

# Box Plot: Age distribution by Diagnosis with Patient Highlight
st.subheader("Age Distribution by Diagnosis")
fig, ax = plt.subplots()
sns.boxplot(data=alzheimers_data, x='Diagnosis', y='Age', palette="Set2")
ax.scatter(patient_data['Diagnosis'].values[0], patient_data['Age'].values[0], color="red", label="Selected Patient")
plt.title("Age Distribution by Diagnosis")
plt.legend()
st.pyplot(fig)

# Patient-Specific Data Summary
st.write("### Selected Patient Summary Statistics")
st.write(patient_data.describe())

# Entire Dataset Statistics
st.write("### Summary Statistics for All Patients")
st.write(alzheimers_data['Alzheimers_Probability_Percentage'].describe())

# Bar Plot of Diagnosis Counts for All Patients
st.subheader("Counts of Alzheimer's Diagnosis (All Patients)")
fig, ax = plt.subplots()
alzheimers_data['Diagnosis'].value_counts().plot(kind='bar', color=['blue', 'orange'], ax=ax)
plt.title("Counts of Alzheimer's Diagnosis (0 = No, 1 = Yes)")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
st.pyplot(fig)

# Chi-square test for significant genetic variant associations with Diagnosis
significant_variants = []
for column in genome_data.select_dtypes(include=['int', 'float']).columns:
    contingency_table = pd.crosstab(alzheimers_data['Diagnosis'], genome_data[column])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    if p < 0.05:
        significant_variants.append(column)

# Display significant genetic variants associated with Alzheimer's
st.write("### Significant Genetic Variants Associated with Alzheimer's Disease")
if significant_variants:
    st.write(", ".join(significant_variants))
else:
    st.write("No significant variants found.")

# Violin Plot for Alzheimer’s Probability by Diagnosis with Patient Highlight
st.subheader("Alzheimer's Probability by Diagnosis")
fig, ax = plt.subplots()
sns.violinplot(x=alzheimers_data['Diagnosis'], y=alzheimers_data['Alzheimers_Probability_Percentage'], palette="Set2")
ax.scatter(patient_data['Diagnosis'].values[0], patient_data['Alzheimers_Probability_Percentage'].values[0], color="red", label="Selected Patient")
plt.title("Alzheimer's Probability by Diagnosis")
plt.xlabel("Diagnosis (0 = No Alzheimer's, 1 = Alzheimer's)")
plt.ylabel("Alzheimer's Probability (%)")
plt.legend()
st.pyplot(fig)

# Model Evaluation
y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy:.2f}")
