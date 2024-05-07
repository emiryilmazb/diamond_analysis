"""
code
..
code output
..
code description (all steps)
..
1 Mean, median, and mode

2 boxplot analysis
3 correlated and uncorrelated data
4 similarity and dissimilarity
5 min-max normalization
6 z-score normalization
7 normalization by decimal scaling
8 mean absolute deviation
9 minkowski distance
10 euclidean distance
11 manhattan distance
12 supremum distance
13 cosine similarity
14 chi-square calculation
15 covariance
16 co-variance
17 binning methods for data smoothing (equi-depth, bin means, bin boundaries)
18 decision tree algorithm
19 naive bayes classifier
20 calssifier evaluation metrics: precision and recall and f-measures
"""
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_poisson_deviance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cityblock, chebyshev

diamonds = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/diamond.csv')
price = diamonds["Price"]
numerical_cols = ['Carat Weight','Price']
categorical_cols = ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']

# Mean
mean_price = price.mean()

# Median
median_price = price.median()

# Mode
mode_price = price.mode()

print("Mean:", mean_price)
print("Median:", median_price)
print("Mode:", mode_price)

numerical_cols = ['Carat Weight','Price']
categorical_cols = ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']

# Create boxplots
plt.figure(figsize=(18, 12))

# Plot numerical columns
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=diamonds[col])
    plt.title(col)

# Plot categorical columns
for i, col in enumerate(categorical_cols, len(numerical_cols) + 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=diamonds[col], palette='Set3', hue=diamonds[col], legend=False)
    plt.title(col)

plt.tight_layout()
plt.show()

correlation_matrix = diamonds[numerical_cols].corr()

print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Columns")
plt.show()

# One-hot encode categorical columns
diamonds_encoded = pd.get_dummies(diamonds)

# Calculate correlation matrix
correlation_matrix = diamonds_encoded.corr()

# Display correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of All Columns")
plt.show()

for col in categorical_cols:
    print("\nFrequency Analysis for", col)
    print(diamonds[col].value_counts())

# Visualization
# Histograms for Numerical Columns
diamonds[numerical_cols].hist()
plt.show()

# Boxplot for Price by Cut
sns.boxplot(x='Cut', y='Price', data=diamonds)
plt.show()

# Scatter plot for Carat Weight vs Price
plt.scatter(diamonds['Carat Weight'], diamonds['Price'])
plt.xlabel('Carat Weight')
plt.ylabel('Price')
plt.title('Scatter Plot of Carat Weight vs Price')
plt.show()

# Extract the numerical data
numerical_data = diamonds[numerical_cols]

# Perform z-score normalization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)

# Display the first few rows of the scaled DataFrame
print(scaled_df.head())

def decimal_scaling(data):
    max_abs = data.abs().max()
    return data / (10 ** len(str(int(max_abs))))

# Apply decimal scaling normalization to the numerical data
scaled_data = numerical_data.apply(decimal_scaling)

# Display the first few rows of the scaled DataFrame
print(scaled_data.head())

def mad(data):
    return data.sub(data.mean()).abs().mean()

# Function to perform MAD normalization
def mad_normalization(data):
    mean = data.mean()
    mean_absolute_deviation = mad(data)
    return (data - mean) / mean_absolute_deviation

# Apply MAD normalization to the numerical data
normalized_data = numerical_data.apply(mad_normalization)

# Display the first few rows of the normalized DataFrame
print(normalized_data.head())

from scipy.spatial.distance import minkowski

# Define two vectors from your dataset
vector_X = diamonds.loc[0, ['Carat Weight', 'Price']].values
vector_Y = diamonds.loc[1, ['Carat Weight', 'Price']].values

# Define the parameter p (for example, p=2 for Euclidean distance)
p = 2

# Calculate the Minkowski distance
minkowski_distance = minkowski(vector_X, vector_Y, p)

print("Minkowski distance between X and Y:", minkowski_distance)

# Calculate the Euclidean distance
euclidean_distance = euclidean(vector_X, vector_Y)

print("Euclidean distance between X and Y:", euclidean_distance)

manhattan_distance = cityblock(vector_X, vector_Y)

print("Manhattan distance between X and Y:", manhattan_distance)

supremum_distance = chebyshev(vector_X, vector_Y)

print("Supremum distance between X and Y:", supremum_distance)

vector_X = vector_X.reshape(1, -1)
vector_Y = vector_Y.reshape(1, -1)

# Calculate cosine similarity
cos_sim = cosine_similarity(vector_X, vector_Y)[0][0]

print("Cosine Similarity between vector A and vector B:", cos_sim)

contingency_table = pd.crosstab(diamonds['Cut'], diamonds['Color'])

# Calculate the chi-square statistic, p-value, degrees of freedom, and expected frequencies
chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2_stat)
print("P-value:", p_val)
print("Degrees of freedom:", dof)
print("Expected frequencies:")
print(expected)

# Calculate covariance between two specific columns
covariance = numerical_data['Carat Weight'].cov(numerical_data['Price'])

print("Covariance between 'Carat Weight' and 'Price':", covariance)