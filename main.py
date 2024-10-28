import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight

# ---------------------------
# Load Alzheimer's clinical dataset
# ---------------------------
alzheimers_data = pd.read_csv("C:\\Users\\Victus\\OneDrive\\Desktop\\alzheimers_disease_data.csv")
alzheimers_data = alzheimers_data.dropna(subset=['PatientID', 'Age', 'Diagnosis'])

# Normalize clinical features
scaler = StandardScaler()
clinical_features = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP']
alzheimers_data[clinical_features] = scaler.fit_transform(alzheimers_data[clinical_features])

# APOE Genotype Distribution
apoe_genotypes = ['e2/e2', 'e2/e3', 'e2/e4', 'e3/e3', 'e3/e4', 'e4/e4']
genotype_probabilities = [0.01, 0.13, 0.03, 0.60, 0.20, 0.03]
np.random.seed(42)
alzheimers_data['APOE_genotype'] = np.random.choice(apoe_genotypes, size=len(alzheimers_data), p=genotype_probabilities)

# Map APOE genotypes to risk scores
apoe_risk_mapping = {'e2/e2': 0, 'e2/e3': 0, 'e3/e3': 1, 'e2/e4': 1, 'e3/e4': 1, 'e4/e4': 2}
alzheimers_data['APOE_risk_score'] = alzheimers_data['APOE_genotype'].map(apoe_risk_mapping)

# ---------------------------
# Genome Data Analysis with SDps
# ---------------------------
genomes_data = pd.read_csv("C:\\Users\\Victus\\OneDrive\\Desktop\\1000-genomes-phase-3_reports_1000_Genomes_phase_3_sample_results.csv")
relevant_genome_columns = ['name', 'heterozygous_variant_count', 'perct_hom_alt_in_snvs']
genomes_data_filtered = genomes_data[relevant_genome_columns].dropna()

# Calculate SDps and PRS based on SDps
genomes_data_filtered['SDps_het_var_count'] = genomes_data_filtered['heterozygous_variant_count'].std()
genomes_data_filtered['SDps_hom_alt_in_snvs'] = genomes_data_filtered['perct_hom_alt_in_snvs'].std()

genomes_data_filtered['PRS_SDps'] = genomes_data_filtered.apply(
    lambda row: row['heterozygous_variant_count'] * row['SDps_het_var_count'] + row['perct_hom_alt_in_snvs'] * row['SDps_hom_alt_in_snvs'],
    axis=1
)

# Define risk category based on PRS threshold
risk_threshold_genome_sdps = genomes_data_filtered['PRS_SDps'].median()
genomes_data_filtered['RiskCategory_SDps'] = genomes_data_filtered['PRS_SDps'].apply(lambda x: 1 if x > risk_threshold_genome_sdps else 0)

# Add SDps risk to Alzheimer's data
alzheimers_data['SDps_risk'] = genomes_data_filtered['RiskCategory_SDps'][:len(alzheimers_data)].values

# ---------------------------
# Random Forest Model on Clinical Data
# ---------------------------
X_clinical = alzheimers_data[clinical_features]
y_clinical = alzheimers_data['Diagnosis'].apply(lambda x: 1 if x == 'Alzheimers' else 0)

X_train_clinical, X_test_clinical, y_train_clinical, y_test_clinical = train_test_split(X_clinical, y_clinical, test_size=0.2, random_state=42)

class_weights_clinical = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_clinical), y= y_train_clinical)
class_weights_dict_clinical = {i: class_weights_clinical[i] for i in range(len(class_weights_clinical))}

rf_clinical = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict_clinical)
rf_clinical.fit(X_train_clinical, y_train_clinical)

y_pred_clinical = rf_clinical.predict(X_test_clinical)
accuracy_clinical = accuracy_score(y_test_clinical, y_pred_clinical)
print(f"Random Forest Accuracy on Clinical Data: {accuracy_clinical * 100:.2f}%")

# ---------------------------
# Logistic Regression Model combining APOE risk and SDps
# ---------------------------
combined_df = pd.DataFrame({
    'APOE_risk_score': alzheimers_data['APOE_risk_score'],
    'SDps_risk': genomes_data_filtered['RiskCategory_SDps'][:len(alzheimers_data)]  # Ensure same length
})

X = combined_df[['APOE_risk_score']]
y = combined_df['SDps_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Model Accuracy: {accuracy * 100:.2f}%')

# Graphs
plt.figure(figsize=(8, 5))
sns.countplot(x='APOE_genotype', data=alzheimers_data, palette='viridis')
plt.title('Distribution of APOE Genotypes')
plt.xlabel('APOE Genotype')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='APOE_risk_score', hue='SDps_risk', data=alzheimers_data, palette='magma')
plt.title('APOE Risk Score vs SDps Risk Category')
plt.xlabel('APOE Risk Score')
plt.ylabel('Count')
plt.legend(title='SDps Risk Category', loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
correlation_matrix = alzheimers_data[clinical_features + ['APOE_risk_score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Clinical Features')
plt.show()

plt.figure(figsize=(10, 6))
genomes_data_filtered['APOE_genotype'] = alzheimers_data['APOE_genotype'][:len(genomes_data_filtered)]
sns.boxplot(x='APOE_genotype', y='PRS_SDps', data=genomes_data_filtered, palette='Set2')
plt.title('Polygenic Risk Score (PRS) Distribution by APOE Genotype')
plt.xlabel('APOE Genotype')
plt.ylabel('PRS (using SDps)')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(genomes_data_filtered['PRS_SDps'], bins=30, color='skyblue', edgecolor='black')
plt.title('Polygenic Risk Score (PRS) Distribution using SDps')
plt.xlabel('PRS (using SDps)')
plt.ylabel('Frequency')
plt.show()

# Create a combined DataFrame for the scatter plot
scatter_df = pd.DataFrame({
    'APOE_risk_score': alzheimers_data['APOE_risk_score'],
    'PRS_SDps': genomes_data_filtered['PRS_SDps'][:len(alzheimers_data)],
    'APOE_genotype': alzheimers_data['APOE_genotype']
})

plt.figure(figsize=(8, 6))
sns.scatterplot(x='APOE_risk_score', y='PRS_SDps', data=scatter_df, hue='APOE_genotype', palette='viridis')
plt.xlabel('APOE Risk Score')
plt.ylabel('PRS (using SDps)')
plt.title('Scatter Plot of APOE Risk Score vs PRS (using SDps)')
plt.legend(title='APOE Genotype', loc='upper right')
plt.show()

# Additional Graphs
if 'APOE_risk_score' in genomes_data_filtered.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(genomes_data_filtered['APOE_risk_score'], bins=20, kde=True, color='blue')
    plt.xlabel('APOE Risk Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of APOE Risk Scores')
    plt.show()
else:
    print("Column 'APOE_risk_score' is missing in genomes_data_filtered.")

if 'PRS_SDps' in genomes_data_filtered.columns and 'APOE_genotype' in genomes_data_filtered.columns:
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='APOE_genotype', y='PRS_SDps', data=genomes_data_filtered, palette='Set2')
    plt.xlabel('APOE Genotype')
    plt.ylabel('PRS (using SDps)')
    plt.title('Violin Plot of PRS (using SDps) by APOE Genotype')
    plt.show()
else:
    print("One or both of the columns 'PRS_SDps' and 'APOE_genotype' are missing in genomes_data_filtered.")

if 'APOE_risk_score' in genomes_data_filtered.columns and 'APOE_genotype' in genomes_data_filtered.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='APOE_genotype', y='APOE_risk_score', data=genomes_data_filtered, palette='Pastel1')
    plt.xlabel('APOE Genotype')
    plt.ylabel('APOE Risk Score')
    plt.title('Box Plot of APOE Risk Scores by APOE Genotype')
    plt.show()
else:
    print("One or both of the columns 'APOE_risk_score' and 'APOE_genotype' are missing in genomes_data_filtered.")