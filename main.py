from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OrdinalEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_poisson_deviance, precision_score, recall_score, f1_score, accuracy_score, jaccard_score, classification_report
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, minkowski, euclidean, hamming, chebyshev
from scipy.spatial import distance
from scipy.stats import chi2_contingency

# Load the dataset
diamonds = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/diamond.csv')
price = diamonds["Price"]

# Identify the numerical and categorical columns
numerical_cols = ['Carat Weight', 'Price']
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

# Boxplot analysis for each numerical column separately
for col in numerical_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(y=diamonds[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Boxplot analysis for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=col, y='Price', data=diamonds)
    plt.title(f'Boxplot of Price by {col}')
    plt.xticks(rotation=45)  # Rotate x labels for better readability
    plt.show()

    correlation_matrix = diamonds[numerical_cols].corr()
df = diamonds[numerical_cols + categorical_cols]
df_encoded = pd.get_dummies(df, columns=categorical_cols)
corr_matrix = df_encoded.corr()
# print("Correlation Matrix:")
# print(corr_matrix)

plt.figure(figsize=(25, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Correlation Matrix of Numerical Columns')
plt.show()

numerical_data = diamonds[numerical_cols]
correlation_matrix = numerical_data.corr()
print(correlation_matrix)

scaler = StandardScaler()
numerical_data_scaled = scaler.fit_transform(numerical_data)

# Calculate Euclidean distance between the first two rows as an example
distance = euclidean(numerical_data_scaled[0], numerical_data_scaled[1])
print(distance)

encoder = OneHotEncoder(sparse=False)
categorical_data_encoded = encoder.fit_transform(diamonds[categorical_cols])

# Calculate Jaccard similarity between the first two rows as an example
similarity = jaccard_score(categorical_data_encoded[0], categorical_data_encoded[1], average='macro')
similarity
# Calculate Hamming distance between the first two rows as an example
hamming_distance = hamming(categorical_data_encoded[0], categorical_data_encoded[1])
print(hamming_distance)

scaler = MinMaxScaler()
diamonds[numerical_cols] = scaler.fit_transform(diamonds[numerical_cols])

# Display the first few rows of the normalized dataset
print(diamonds.head())

scaler = StandardScaler()
diamonds[numerical_cols] = scaler.fit_transform(diamonds[numerical_cols])

# Display the first few rows of the normalized dataset
print(diamonds.head())

for col in numerical_cols:
    max_abs = diamonds[col].abs().max()
    diamonds[col] = diamonds[col] / (10 ** len(str(int(max_abs))))

# Display the first few rows of the normalized dataset
print(diamonds.head())

for col in numerical_cols:
    mean_val = diamonds[col].mean()
    abs_dev = diamonds[col].sub(mean_val).abs().mean()
    diamonds[col] = diamonds[col].sub(mean_val).div(abs_dev)

# Display the first few rows of the normalized dataset
print(diamonds.head())

label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    diamonds[col] = label_encoders[col].fit_transform(diamonds[col])

# Calculate Minkowski distance
point1 = diamonds.loc[0, numerical_cols + categorical_cols].values
point2 = diamonds.loc[1, numerical_cols + categorical_cols].values

minkowski_distance = minkowski(point1, point2, p=2)  # p=2 for Euclidean distance, p=1 for Manhattan distance

print("Minkowski distance between the two points:", minkowski_distance)

# Select two points (rows) for demonstration
point1 = diamonds.loc[0, numerical_cols].values
point2 = diamonds.loc[1, numerical_cols].values

# Calculate Euclidean distance
euclidean_distance = np.linalg.norm(point1 - point2)

print("Euclidean distance between the two points:", euclidean_distance)

minkowski_distance = pdist(diamonds[numerical_cols], 'minkowski', p=3)
print("Minkowski Distance:")
print(minkowski_distance)

supremum_distance = chebyshev(diamonds[numerical_cols].iloc[0], diamonds[numerical_cols].iloc[1])
print("Supremum Distance between first two rows:")
print(supremum_distance)

cosine_sim = cosine_similarity(diamonds[numerical_cols])[0][1]
print("Cosine Similarity between first two rows:")
print(cosine_sim)

contingency_table = pd.crosstab(diamonds['Cut'], diamonds['Color'])

# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square value:", chi2)
print("p-value:", p)

covariance_matrix = diamonds[numerical_cols].cov()
print("Covariance Matrix:")
print(covariance_matrix)


covariance = diamonds[numerical_cols[0]].cov(diamonds[numerical_cols[1]])
print("Covariance between Carat Weight and Price:")
print(covariance)

def equi_depth_binning(data, num_bins):
    data_sorted = sorted(data)
    bin_size = len(data) // num_bins
    bins = [data_sorted[i:i+bin_size] for i in range(0, len(data), bin_size)]
    return bins

# Example usage:
equi_depth_bins = equi_depth_binning(diamonds['Carat Weight'], num_bins=5)
print("Equi-depth Bins:")
print(equi_depth_bins)


def bin_means_binning(data, num_bins):
    bin_size = len(data) // num_bins
    bins = [data[i:i+bin_size].mean() for i in range(0, len(data), bin_size)]
    return bins

# Example usage:
bin_means_bins = bin_means_binning(diamonds['Price'], num_bins=5)
print("Bin Means Bins:")
print(bin_means_bins)


def bin_boundaries_binning(data, num_bins):
    bin_size = len(data) // num_bins
    bins = [data[i:i+bin_size].iloc[-1] for i in range(0, len(data), bin_size)]
    return bins

# Example usage:
bin_boundaries_bins = bin_boundaries_binning(diamonds['Price'], num_bins=5)
print("Bin Boundaries Bins:")
print(bin_boundaries_bins)


# Define features and target variable
X = diamonds.drop("Price", axis=1)  # Adjust column name if needed
y = diamonds["Price"]  # Adjust column name if needed

# Alternatively, use the following if 'Price' column is not found as expected:
# X = diamonds.drop(diamonds.columns[-1], axis=1)  # Assuming 'Price' is the last column
# y = diamonds[diamonds.columns[-1]]

# Define numerical and categorical columns
numerical_cols = ['Carat Weight']
categorical_cols = ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Create a pipeline with preprocessing and the regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model (optional)
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
y_binned = discretizer.fit_transform(y.values.reshape(-1, 1)).flatten()

# Define numerical and categorical columns
numerical_cols = ['Carat Weight']
categorical_cols = ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
    ])

# Create a pipeline with preprocessing and the Naive Bayes classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict the target for the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Regression Metrics
print("Regression Metrics:")
print("===================")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print()
# Classification Metrics
print("Classification Metrics:")
print("=======================")
print(f"Accuracy: {accuracy:.2f}")
print()
print("Detailed Classification Report:")
print("-------------------------------")
print(report)

