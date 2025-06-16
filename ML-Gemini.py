# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit # For basic time-series split if full walk-forward is complex
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb # Or import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Load datasets
try:
    train_df = pd.read_csv('train/train.csv')
    test_df = pd.read_csv('test.csv')
    # Assuming transactions.csv is now available
    transactions_df = pd.read_csv('train/transactions.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Ensure 'train.csv', 'test.csv', and 'transactions.csv' are in the same directory.")
    exit()

# Display initial information and sample rows
print("\n--- Train Data Info ---")
train_df.info()
print("\n--- Train Data Head ---")
print(train_df.head())

print("\n--- Test Data Info ---")
test_df.info()
print("\n--- Test Data Head ---")
print(test_df.head())

print("\n--- Transactions Data Info ---")
transactions_df.info()
print("\n--- Transactions Data Head ---")
print(transactions_df.head())

# %%
def preprocess_and_feature_engineer(df, is_train=True, historical_data=None, transactions_data=None):
    """
    Performs date preprocessing and feature engineering.
    historical_data is used to calculate aggregates for the test set.
    transactions_data is used to merge transaction-level features.
    """
    df['doj'] = pd.to_datetime(df['doj'])

    # Extract temporal features from doj
    df['year'] = df['doj'].dt.year
    df['month'] = df['doj'].dt.month
    df['day'] = df['doj'].dt.day
    df['day_of_week'] = df['doj'].dt.dayofweek
    df['day_of_year'] = df['doj'].dt.dayofyear
    df['week_of_year'] = df['doj'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['doj'].dt.dayofweek >= 5).astype(int) # Saturday (5) or Sunday (6)

    # Placeholder for public holidays (requires external data)
    # For a real hackathon, this would involve loading a holiday calendar
    # and creating features like 'is_holiday', 'days_until_holiday', etc.

    # Combine srcid and destid to create a unique route identifier for aggregation
    df['route_id'] = df['srcid'].astype(str) + '_' + df['destid'].astype(str)

    # --- Integrate transactions data ---
    if transactions_data is not None:
        transactions_data['doj'] = pd.to_datetime(transactions_data['doj'])
        transactions_data['doi'] = pd.to_datetime(transactions_data['doi'])

        # Example: Aggregate transaction data to the same granularity as train/test_df
        # This part needs to be carefully designed to avoid data leakage.
        # For a 15-day prediction window, for a given doj in test_df, you can only use
        # transaction data up to (doj - 15 days).
        # This often involves a complex rolling join or pre-aggregation.
        
        # For now, let's assume we are creating features from transactions_data
        # that are relevant for the 'doj' in the main dataframe.
        # This is a placeholder and needs to be refined based on actual data structure and leakage prevention.
        
        # Example: Calculate average cumsum_seatcount and cumsum_searchcount for each route_id, doj
        # This is a simplified aggregation. In reality, you'd need to consider the 'doi' and 'dbd'
        # to ensure you're only using data available 15 days prior to 'doj'.
        transactions_agg = transactions_data.groupby(['doj', 'srcid', 'destid']).agg(
            avg_cumsum_seatcount=('cumsum_seatcount', 'mean'),
            avg_cumsum_searchcount=('cumsum_searchcount', 'mean'),
            # You might also want to capture the last known dbd, or average dbd
            # avg_dbd=('dbd', 'mean')
        ).reset_index()

        df = pd.merge(df, transactions_agg, on=['doj', 'srcid', 'destid'], how='left')

        # Also, directly use region and tier features from transactions_data
        # Assuming srcid_region, destid_region, srcid_tier, destid_tier are consistent per srcid/destid
        region_tier_map_src = transactions_data[['srcid', 'srcid_region', 'srcid_tier']].drop_duplicates().set_index('srcid')
        region_tier_map_dest = transactions_data[['destid', 'destid_region', 'destid_tier']].drop_duplicates().set_index('destid')

        df['srcid_region'] = df['srcid'].map(region_tier_map_src['srcid_region'])
        df['srcid_tier'] = df['srcid'].map(region_tier_map_src['srcid_tier'])
        df['destid_region'] = df['destid'].map(region_tier_map_dest['destid_region'])
        df['destid_tier'] = df['destid'].map(region_tier_map_dest['destid_tier'])

        # Fill any NaNs in new features (e.g., for routes not in transactions_data or future dates)
        for col in ['avg_cumsum_seatcount', 'avg_cumsum_searchcount']:
            df[col].fillna(0, inplace=True) # Or a more appropriate fill value like mean/median

        # Convert new categorical features to category type
        for col in ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']:
            df[col] = df[col].astype('category')

    if is_train:
        # Calculate route-specific aggregates from training data
        route_aggregates = df.groupby('route_id')['final_seatcount'].agg(
            route_mean_seatcount='mean',
            route_median_seatcount='median',
            route_std_seatcount='std'
        ).reset_index()
        df = pd.merge(df, route_aggregates, on='route_id', how='left')

        # Create lagged features (for training data, fill NaNs appropriately)
        df = df.sort_values(by=['route_id', 'doj'])
        df['final_seatcount_lag_7'] = df.groupby('route_id')['final_seatcount'].shift(7)
        df['final_seatcount_lag_14'] = df.groupby('route_id')['final_seatcount'].shift(14)

        # Rolling statistics (for training data)
        df['rolling_mean_7_days'] = df.groupby('route_id')['final_seatcount'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df['rolling_std_7_days'] = df.groupby('route_id')['final_seatcount'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )

        # Fill NaNs created by lagging/rolling (e.g., with route mean or 0)
        df['final_seatcount_lag_7'].fillna(df['route_mean_seatcount'], inplace=True)
        df['final_seatcount_lag_14'].fillna(df['route_mean_seatcount'], inplace=True)
        df['rolling_mean_7_days'].fillna(df['route_mean_seatcount'], inplace=True)
        df['rolling_std_7_days'].fillna(0, inplace=True)

        return df, route_aggregates
    else:
        # For test data, use aggregates from the historical_data (training data)
        df = pd.merge(df, historical_data, on='route_id', how='left')

        # For test data, lagged features and rolling statistics need careful handling.
        # This is a simplification; a more robust solution would involve iteratively
        # predicting and then using those predictions to generate subsequent lags,
        # or using features that are known 15 days in advance.
        df['final_seatcount_lag_7'] = df['route_mean_seatcount'] # Placeholder
        df['final_seatcount_lag_14'] = df['route_mean_seatcount'] # Placeholder
        df['rolling_mean_7_days'] = df['route_mean_seatcount'] # Placeholder
        df['rolling_std_7_days'] = df['route_std_seatcount'].fillna(0) # Placeholder

        # Handle routes in test_df not present in train_df (if any)
        # Corrected: Use the column names as they exist in historical_data
        for col_name in ['route_mean_seatcount', 'route_median_seatcount', 'route_std_seatcount']:
            df[col_name].fillna(historical_data[col_name].mean(), inplace=True)

        return df

# Process training data
train_df_processed, route_aggregates_for_test = preprocess_and_feature_engineer(train_df.copy(), is_train=True, transactions_data=transactions_df.copy())

# Process test data
# For test data, ensure transactions_data is appropriately filtered to avoid data leakage.
# For a 15-day prediction window, for a test_doj, only use transactions where doi <= (test_doj - 15 days).
# This is a complex step and simplified here.
test_df_processed = preprocess_and_feature_engineer(test_df.copy(), is_train=False, historical_data=route_aggregates_for_test, transactions_data=transactions_df.copy())

# Display processed data head and check for NaNs
print("\n--- Processed Train Data Head ---")
print(train_df_processed.head())
print("\n--- Processed Train Data NaNs ---")
print(train_df_processed.isnull().sum())

print("\n--- Processed Test Data Head ---")
print(test_df_processed.head())
print("\n--- Processed Test Data NaNs ---")
print(test_df_processed.isnull().sum())

# Define features and target
features = [col for col in train_df_processed.columns if col not in ['doj', 'final_seatcount', 'route_key', 'route_id']]
target = 'final_seatcount'

# Convert categorical features for LightGBM
categorical_features = ['srcid', 'destid', 'month', 'day_of_week', 'is_weekend',
                        'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier'] # Added new categorical features
for col in categorical_features:
    if col in features: # Check if feature exists after processing
        train_df_processed[col] = train_df_processed[col].astype('category')
        test_df_processed[col] = test_df_processed[col].astype('category')

print(f"\nFeatures used for training: {features}")

# --- Model Training (Simplified Train/Validation Split for demonstration) ---
# For a real hackathon, implement full Walk-Forward Validation as discussed.
# Here, we'll simulate a simple time-based split for quick demonstration.
# Sort data by date for time-series split
train_df_processed = train_df_processed.sort_values(by='doj').reset_index(drop=True)

# Define a cutoff date for validation
train_cutoff_date = pd.to_datetime('2024-07-01') # Example: Use data before July 2024 for training

X_train_val = train_df_processed[train_df_processed['doj'] < train_cutoff_date][features]
y_train_val = train_df_processed[train_df_processed['doj'] < train_cutoff_date][target]
X_val = train_df_processed[train_df_processed['doj'] >= train_cutoff_date][features]
y_val = train_df_processed[train_df_processed['doj'] >= train_cutoff_date][target]

# Initialize LightGBM Regressor Model
lgb_model = lgb.LGBMRegressor(objective='regression_l1', # MAE objective
                              metric='mae',
                              n_estimators=1000,
                              learning_rate=0.05,
                              num_leaves=31,
                              max_depth=-1,
                              min_child_samples=20,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              random_state=42,
                              n_jobs=-1)

print("\n--- Training LightGBM Model ---")
lgb_model.fit(X_train_val, y_train_val,
              eval_set=[(X_val, y_val)],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(100, verbose=False)], # Early stopping to prevent overfitting
              categorical_feature=[col for col in categorical_features if col in features])

# Evaluate on validation set
val_predictions = lgb_model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_predictions)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"\nValidation MAE: {val_mae:.2f}")
print(f"Validation RMSE: {val_rmse:.2f}")

# --- Final Model Training on Full Training Data ---
print("\n--- Training Final LightGBM Model on Full Data ---")
final_lgb_model = lgb.LGBMRegressor(objective='regression_l1', # MAE objective
                                    metric='mae',
                                    n_estimators=lgb_model.best_iteration_, # Use best iteration from validation
                                    learning_rate=0.05,
                                    num_leaves=31,
                                    max_depth=-1,
                                    min_child_samples=20,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    random_state=42,
                                    n_jobs=-1)

final_lgb_model.fit(train_df_processed[features], train_df_processed[target],
                    categorical_feature=[col for col in categorical_features if col in features])

# --- Generate Predictions for Test Set ---
print("\n--- Generating Predictions for Test Set ---")
test_predictions = final_lgb_model.predict(test_df_processed[features])

# Ensure predictions are non-negative and round to nearest integer if required
test_predictions[test_predictions < 0] = 0
test_predictions = np.round(test_predictions).astype(int) # Assuming integer seat counts

# %%
# Create submission DataFrame
submission_df = pd.DataFrame({
    'route_key': test_df_processed['route_key'],
    'final_seatcount': test_predictions
})

# Display submission file head
print("\n--- Submission File Head ---")
print(submission_df.head())

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully.")


