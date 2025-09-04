import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

print("Loading data")

train_data = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test_data = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")

print("=== STEP 2: DATA PREPROCESSING ===")

# Create a copy for processing
train_processed = train_data.copy()
test_processed = test_data.copy()

# 1. Handle missing values
print("Handling missing values...")

# For sale_nbr (numerical), fill with median
train_processed['sale_nbr'].fillna(train_processed['sale_nbr'].median(), inplace=True)
test_processed['sale_nbr'].fillna(train_processed['sale_nbr'].median(), inplace=True)

# For subdivision and submarket (categorical), fill with 'Unknown'
train_processed['subdivision'].fillna('Unknown', inplace=True)
test_processed['subdivision'].fillna('Unknown', inplace=True)
train_processed['submarket'].fillna('Unknown', inplace=True)
test_processed['submarket'].fillna('Unknown', inplace=True)

print("Missing values handled.")

# 2. Feature Engineering
print("Creating new features...")

# Date features
train_processed['sale_date'] = pd.to_datetime(train_processed['sale_date'])
test_processed['sale_date'] = pd.to_datetime(test_processed['sale_date'])

train_processed['sale_year'] = train_processed['sale_date'].dt.year
train_processed['sale_month'] = train_processed['sale_date'].dt.month
train_processed['sale_quarter'] = train_processed['sale_date'].dt.quarter

test_processed['sale_year'] = test_processed['sale_date'].dt.year
test_processed['sale_month'] = test_processed['sale_date'].dt.month
test_processed['sale_quarter'] = test_processed['sale_date'].dt.quarter

# Age of house at sale
train_processed['house_age'] = train_processed['sale_year'] - train_processed['year_built']
test_processed['house_age'] = test_processed['sale_year'] - test_processed['year_built']

# Years since renovation (0 if never renovated)
train_processed['years_since_reno'] = np.where(
    train_processed['year_reno'] > 0,
    train_processed['sale_year'] - train_processed['year_reno'],
    train_processed['house_age']
)
test_processed['years_since_reno'] = np.where(
    test_processed['year_reno'] > 0,
    test_processed['sale_year'] - test_processed['year_reno'],
    test_processed['house_age']
)

# Total bathrooms
train_processed['total_baths'] = (train_processed['bath_full'] + 
                                 train_processed['bath_3qtr'] * 0.75 + 
                                 train_processed['bath_half'] * 0.5)
test_processed['total_baths'] = (test_processed['bath_full'] + 
                                test_processed['bath_3qtr'] * 0.75 + 
                                test_processed['bath_half'] * 0.5)

# Price per sqft (only for training data)
train_processed['price_per_sqft'] = train_processed['sale_price'] / train_processed['sqft']

# Total value (land + improvement)
train_processed['total_val'] = train_processed['land_val'] + train_processed['imp_val']
test_processed['total_val'] = test_processed['land_val'] + test_processed['imp_val']

# Total view score (sum of all view features)
view_cols = [col for col in train_processed.columns if col.startswith('view_')]
train_processed['total_views'] = train_processed[view_cols].sum(axis=1)
test_processed['total_views'] = test_processed[view_cols].sum(axis=1)

# Garage + basement sqft
train_processed['total_extra_sqft'] = train_processed['garb_sqft'] + train_processed['gara_sqft']
test_processed['total_extra_sqft'] = test_processed['garb_sqft'] + test_processed['gara_sqft']

print("New features created.")

# 3. Encode categorical variables
print("Encoding categorical variables...")

categorical_cols = ['sale_warning', 'join_status', 'city', 'zoning', 'subdivision', 'submarket']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined data to ensure consistent encoding
    combined_data = pd.concat([train_processed[col], test_processed[col]], axis=0)
    le.fit(combined_data.astype(str))
    
    train_processed[col + '_encoded'] = le.transform(train_processed[col].astype(str))
    test_processed[col + '_encoded'] = le.transform(test_processed[col].astype(str))
    
    label_encoders[col] = le

print("Categorical variables encoded.")

# 4. Select features for modeling
print("Selecting features...")

# Features to exclude
exclude_cols = ['id', 'sale_date', 'sale_price', 'price_per_sqft'] + categorical_cols

# Get all feature columns
feature_cols = [col for col in train_processed.columns if col not in exclude_cols]

print(f"Selected {len(feature_cols)} features for modeling:")
print(feature_cols[:10], "... (showing first 10)")

# Prepare final datasets
X_train = train_processed[feature_cols]
y_train = train_processed['sale_price']
X_test = test_processed[feature_cols]
test_ids = test_processed['id']

print(f"\nFinal shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")

# 5. Check for any remaining issues
print(f"\nData quality check:")
print(f"X_train missing values: {X_train.isnull().sum().sum()}")
print(f"X_test missing values: {X_test.isnull().sum().sum()}")
print(f"y_train missing values: {y_train.isnull().sum()}")

# Save processed data
print("\nSaving processed data...")
X_train.to_csv('X_train_processed.csv', index=False)
y_train.to_csv('y_train_processed.csv', index=False)
X_test.to_csv('X_test_processed.csv', index=False)
test_ids.to_csv('test_ids.csv', index=False)

print("Step 2 completed successfully!")
print("\nNext step will involve building prediction interval models.")