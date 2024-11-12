import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler  # Fixed: Import for StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
try:
    alzheimers_data = pd.read_csv('/home/r1ddh1/2nd_year/dpel_project_things/Gene Insights/GENE_INSIGHTS/Datasets/alzheimers_disease_data.csv')
    genome_data = pd.read_csv('/home/r1ddh1/2nd_year/dpel_project_things/Gene Insights/GENE_INSIGHTS/Datasets/1000-genomes-phase-3_reports_1000_Genomes_phase_3_sample_results.csv')
except FileNotFoundError:
    st.error("File not found. Please check the file paths.")
    st.stop()

# Clean the data by removing NaN values
alzheimers_data.dropna(inplace=True)
genome_data.dropna(inplace=True)

# Ensure 'PatientID' exists
if 'PatientID' not in alzheimers_data.columns:
    alzheimers_data['PatientID'] = range(1, len(alzheimers_data) + 1)

# Feature engineering: Risk factors and health scores
apoe_genotypes = ['e2/e2', 'e2/e3', 'e2/e4', 'e3/e3', 'e3/e4', 'e4/e4']
genotype_probabilities = [0.01, 0.13, 0.03, 0.60, 0.20, 0.03]
apoe_risk_mapping = {'e2/e2': 0, 'e2/e3': 0, 'e3/e3': 1, 'e2/e4': 1, 'e3/e4': 1, 'e4/e4': 2}
alzheimers_data['APOE_genotype'] = np.random.choice(apoe_genotypes, size=len(alzheimers_data), p=genotype_probabilities)
alzheimers_data['APOE_risk_score'] = alzheimers_data['APOE_genotype'].map(apoe_risk_mapping)

# Feature engineering: Additional features based on health and lifestyle
alzheimers_data['CardioMentalScore'] = (alzheimers_data['SystolicBP'] + alzheimers_data['DiastolicBP']) / 2
alzheimers_data['Age'] = alzheimers_data['Age'].astype(float)
alzheimers_data['BMI'] = alzheimers_data['BMI'].astype(float)

# Check if all features are available in the data
features = ['Age', 'BMI', 'CardioMentalScore', 'APOE_risk_score', 'Hypertension', 'Diabetes']
for feature in features:
    if feature not in alzheimers_data.columns:
        st.error(f"Missing feature: {feature}")
        st.stop()

target = 'Diagnosis'  # Assuming binary target: 1 for Alzheimer's, 0 for no Alzheimer's

# Prepare data for model training
X = alzheimers_data[features]
y = alzheimers_data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the Random Forest model pipeline
rf_model = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])

# Train and evaluate the model
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Set up Streamlit dashboard
st.title("Alzheimer's Probability Diagnosis Dashboard")
st.write("This dashboard uses machine learning to predict the probability of Alzheimer's based on genetic and lifestyle data.")

# Patient-specific data
selected_patient = st.selectbox("Select Patient ID", alzheimers_data['PatientID'].unique())
patient_data = alzheimers_data[alzheimers_data['PatientID'] == selected_patient]
probability = rf_model.predict_proba(patient_data[features])[:, 1][0] * 100  # Predicted probability as percentage

# Display patient's Alzheimer's probability (Interactive Gauge)
st.subheader(f"Alzheimer's Probability for Patient {selected_patient}")
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability,
    title={'text': "Alzheimer's Probability (%)"},
    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "purple"}},
    domain={'x': [0, 1], 'y': [0, 1]}
))
fig.update_layout(width=600, height=400)  # Adjusting gauge size
st.plotly_chart(fig)

# Visualizing the overall probability distribution (Interactive Histogram)
st.subheader("Alzheimer's Diagnosis Distribution for All Patients")
fig = px.histogram(alzheimers_data, x='Diagnosis', nbins=2, title="Distribution of Alzheimer's Diagnosis (0 = No, 1 = Yes)",
                   labels={"Diagnosis": "Alzheimer's Diagnosis"})
fig.update_layout(width=800, height=500)  # Adjusting histogram size
st.plotly_chart(fig)

# Interactive Feature Correlation Heatmap (Removing negative correlations)
st.subheader("Feature Correlation Heatmap (With Alzheimer's Diagnosis)")
corr_matrix = alzheimers_data[features + ['Diagnosis']].corr()
corr_matrix = corr_matrix.applymap(lambda x: max(x, 0))  # Setting negative values to zero

fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis', title="Feature Correlation Heatmap")
fig.update_layout(width=800, height=600)  # Adjusting heatmap size
st.plotly_chart(fig)

# Feature importance from Random Forest model (Interactive Bar Chart)
st.subheader("Random Forest Feature Importance")
feature_importance = rf_model.named_steps['rf'].feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

fig = px.bar(importance_df, x='Feature', y='Importance', title="Random Forest Feature Importance", 
             labels={"Feature": "Features", "Importance": "Importance"})
fig.update_layout(width=800, height=500)  # Adjusting bar chart size
st.plotly_chart(fig)

# Display model accuracy
st.write(f"### Model Accuracy: {accuracy:.2f}")

# Boxplot for Age and BMI based on Diagnosis (Interactive)
st.subheader("Boxplot for Age and BMI Based on Diagnosis")
fig_age = px.box(alzheimers_data, x="Diagnosis", y="Age", title="Age Distribution Based on Alzheimer's Diagnosis")
fig_bmi = px.box(alzheimers_data, x="Diagnosis", y="BMI", title="BMI Distribution Based on Alzheimer's Diagnosis")
fig_age.update_layout(width=800, height=500)  # Adjusting boxplot size
fig_bmi.update_layout(width=800, height=500)
st.plotly_chart(fig_age)
st.plotly_chart(fig_bmi)

# ROC Curve
st.subheader("Receiver Operating Characteristic (ROC) Curve")
fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
fig_roc = go.Figure(data=[
    go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='darkblue')),
    go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='red', dash='dash'), name='Random Guess')
])
fig_roc.update_layout(title=f"ROC Curve (AUC = {roc_auc:.2f})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      width=800, height=500)  # Adjusting ROC size
st.plotly_chart(fig_roc)

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm = plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig_cm)

