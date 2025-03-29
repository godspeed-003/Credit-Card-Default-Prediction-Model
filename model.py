from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# First, analyze missing values
def analyze_missing(df):
    missing = df.isnull().sum() / len(df) * 100
    print("Percentage of missing values:")
    print(missing[missing > 0].sort_values(ascending=False))

def preprocess_data(df):
    """
    Preprocess the data by handling missing values and scaling
    """
    df_processed = df.copy()
    
    # Handle missing values
    # For transaction attributes
    transaction_cols = [col for col in df_processed.columns if col.startswith('transaction_')]
    df_processed[transaction_cols] = df_processed[transaction_cols].fillna(0)
    
    # For onus attributes
    onus_cols = [col for col in df_processed.columns if col.startswith('onus_')]
    for col in onus_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        df_processed[f"{col}_missing"] = df_processed[col].isnull().astype(int)
    
    # For bureau attributes
    bureau_cols = [col for col in df_processed.columns if col.startswith('bureau') and not col.startswith('bureau_enquiry')]
    for col in bureau_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # For bureau enquiry attributes
    enquiry_cols = [col for col in df_processed.columns if col.startswith('bureau_enquiry')]
    df_processed[enquiry_cols] = df_processed[enquiry_cols].fillna(0)
    
    return df_processed

# Load data
print("Loading data...")
validation_df = pd.read_csv(r"D:\convulation\validation_data_to_be_shared\validation_data_to_be_shared.csv")
data_df = pd.read_csv(r"D:\convulation\Dev_data_to_be_shared\Dev_data_to_be_shared.csv")

# Analyze missing values
print("Missing values analysis for development data:")
analyze_missing(data_df)
print("\nMissing values analysis for validation data:")
analyze_missing(validation_df)

# Preprocess both datasets
print("\nPreprocessing data...")
data_df_processed = preprocess_data(data_df)
validation_df_processed = preprocess_data(validation_df)

# Split development data
print("\nSplitting training data...")
X = data_df_processed.drop(['account_number', 'bad_flag'], axis=1)
y = data_df_processed['bad_flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with basic configuration
print("\nTraining XGBoost model...")
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    n_estimators=100,  # Fixed number of trees
    learning_rate=0.1,
    max_depth=6
)

# Simple fit without early stopping
model.fit(X_train_scaled, y_train)

# Get probabilities and evaluate
print("\nEvaluating model performance...")
train_probs = model.predict_proba(X_train_scaled)[:, 1]
test_probs = model.predict_proba(X_test_scaled)[:, 1]

print(f"Train AUC: {roc_auc_score(y_train, train_probs):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, test_probs):.4f}")

# Feature importance analysis
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(importance_df.head(10))

# Prepare validation data and make predictions
print("\nMaking predictions on validation data...")
validation_prepared = validation_df_processed.drop(['account_number'], axis=1)
validation_prepared_scaled = scaler.transform(validation_prepared)
predictions = model.predict_proba(validation_prepared_scaled)[:, 1]

# Create and save submission
submission = pd.DataFrame({
    'account_number': validation_df['account_number'],
    'predicted_probability': predictions
})

submission.to_csv('credit_card_predictions.csv', index=False)
print("\nSubmission file has been created: credit_card_predictions.csv")

print("\nPrediction statistics:")
print(submission['predicted_probability'].describe())