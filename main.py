import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

alzheimers_df = pd.read_csv('C:\\Users\\Victus\\OneDrive\\Desktop\\alzheimers_disease_data.csv')
genomes_df = pd.read_csv('C:\\Users\\Victus\\OneDrive\\Desktop\\1000-genomes-phase-3_reports_1000_Genomes_phase_3_sample_results.csv')

# 1. Data Cleaning
# Check for missing values in Alzheimer's data
print("Missing values in Alzheimer's dataset:")
print(alzheimers_df.isnull().sum())

# Drop rows with missing values in Alzheimer's data
alzheimers_df_cleaned = alzheimers_df.dropna()

# Check for missing values in Genomes data
print("\nMissing values in Genomes dataset:")
print(genomes_df.isnull().sum())

# Drop rows with missing values in Genomes data
genomes_df_cleaned = genomes_df.dropna()

# Validate that the data is correctly labeled (e.g., check unique values in 'Diagnosis')
print("\nUnique values in Diagnosis column:")
print(alzheimers_df_cleaned['Diagnosis'].unique())

# 2. Features Calculation
# Example of feature calculation (you can add any relevant calculations)
if 'Hemoglobin' in alzheimers_df_cleaned.columns and 'Hematocrit' in alzheimers_df_cleaned.columns:
    alzheimers_df_cleaned['MCHC'] = alzheimers_df_cleaned['Hemoglobin'] / alzheimers_df_cleaned['Hematocrit'] * 100  # Mean corpuscular hemoglobin concentration

# Normalization of features (example)
features_to_normalize = ['Age']  # Add any other relevant features
for feature in features_to_normalize:
    alzheimers_df_cleaned[feature] = (alzheimers_df_cleaned[feature] - alzheimers_df_cleaned[feature].mean()) / alzheimers_df_cleaned[feature].std()

# 3. Exploratory Data Analysis
# Visualization of distributions of each parameter (e.g., histograms, box plots)
plt.figure(figsize=(10, 6))
sns.histplot(alzheimers_df_cleaned['Age'], bins=20, kde=True)
plt.title('Age Distribution of Alzheimer\'s Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=alzheimers_df_cleaned, x='Diagnosis', y='Age', palette='Set3')
plt.title('Box Plot of Age by Diagnosis Type')
plt.xlabel('Diagnosis')
plt.ylabel('Age')
plt.grid()
plt.show()

# Correlation Matrix
# Select only numeric columns for correlation analysis
numeric_cols = alzheimers_df_cleaned.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix (Alzheimer\'s Data)')
plt.show()

# 4. Identifying Homozygous Alternate Variants
# Assuming 'GenomicVariant' is a relevant column in genomes data that indicates variants
if 'GenomicVariant' in genomes_df_cleaned.columns:
    homozygous_variants = genomes_df_cleaned[genomes_df_cleaned['GenomicVariant'].str.contains('Homozygous', na=False)]
    print(f"\nIdentified Homozygous Alternate Variants:\n{homozygous_variants}")

# 5. Identifying Biomarkers and Therapeutic Targets
# Association Tests (Example: Chi-squared test)
if 'Diagnosis' in alzheimers_df_cleaned.columns and 'Variant' in genomes_df_cleaned.columns:
    contingency_table = pd.crosstab(alzheimers_df_cleaned['Diagnosis'], genomes_df_cleaned['Variant'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-squared test results: chi2={chi2}, p-value={p}")

# Gene Prioritization: Assuming a DataFrame of genes and their scores exists
# Placeholder for prioritization based on a score column
# Example: genes_df = pd.DataFrame({'Gene': [...], 'Score': [...]})
# top_genes = genes_df.nlargest(10, 'Score')
# print(f"Top 10 Genes:\n{top_genes}")

# Pathway Analysis: Placeholder for potential pathway analysis
# This would typically require additional libraries or APIs (e.g., KEGG, Reactome)
# Example: pathway_data = get_pathway_data(top_genes)

# Display basic statistics for both datasets
print("\nAlzheimer's Data Statistics:")
print(alzheimers_df_cleaned.describe())
print("\nGenomes Data Statistics:")
print(genomes_df_cleaned.describe())
