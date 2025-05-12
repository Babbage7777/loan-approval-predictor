import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from IPython.display import display
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def descriptive_analysis(df, title="Descriptive Statistics"):
    """Enhanced descriptive analysis with dynamic visualizations"""
    print(f"\n{title}\n" + "-" * len(title))
    
    # Statistics table
    stats = df.describe(percentiles=[.01, .25, .5, .75, .99]).transpose()
    stats['missing'] = df.isna().sum()
    
    # Only calculate skewness and kurtosis for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        stats['skewness'] = df[numeric_cols].skew()
        stats['kurtosis'] = df[numeric_cols].kurt()
    
    stats['zeros'] = (df == 0).sum()
    
    pd.options.display.float_format = '{:m.2f}'.format
    print("Extended Statistics Summary:")
    display(stats.style.background_gradient(cmap='Blues', subset=['mean', 'std', 'skewness', 'kurtosis']))
    
    # Visualization setup
    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    
    # 1. Numerical distributions
    if not numerical_cols.empty:
        print("\nNumerical Features Analysis:")
        n_num = len(numerical_cols)
        rows = (n_num + 2) // 3
        plt.figure(figsize=(18, 6*rows))
        
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(rows, 3, i)
            sns.histplot(df[col], kde=True, color='teal', bins=30)
            plt.title(f'{col} Distribution', fontsize=10)
        
        plt.tight_layout()
        plt.suptitle(f"{title} - Numerical Distributions", y=1.02, fontsize=14)
        plt.show()
        
        # 2. Boxplots
        plt.figure(figsize=(14, 14))
        sns.boxplot(data=df[numerical_cols], palette='viridis')
        plt.xticks(rotation=45)
        plt.title(f"{title} - Numerical Features Spread", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # 3. Correlation matrix
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 8))
            corr = df[numerical_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, annot=True, cmap='coolwarm', mask=mask, 
                        fmt=".2f", linewidths=.5)
            plt.title(f"{title} - Correlation Matrix", fontsize=14)
            plt.show()
    
    # 4. Categorical analysis
    if not categorical_cols.empty:
        print("\nCategorical Features Analysis:")
        n_cat = len(categorical_cols)
        rows = (n_cat + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(18, 5*rows))
        axes = axes.flatten() if rows > 1 else [axes]
        
        for i, col in enumerate(categorical_cols):
            counts = df[col].value_counts(normalize=True)
            counts.plot(kind='bar', ax=axes[i], color='salmon', edgecolor='black')
            axes[i].set_title(f'{col} Distribution', fontsize=12)
            axes[i].set_ylabel('Percentage')
            axes[i].tick_params(axis='x', rotation=45)
            for p in axes[i].patches:
                height = p.get_height()
                axes[i].text(p.get_x() + p.get_width()/2., height + 0.01,
                            f'{height:.1%}', ha='center', fontsize=8)
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f"{title} - Categorical Distributions", y=1.02, fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # 5. Missing values
    if df.isnull().any().any():
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Missing Values Distribution', fontsize=14)
        plt.show()
    else:
        print("\nNo missing values found in the dataset.")

    return stats

def handle_missing_values(df, numerical_cols, categorical_cols, n_neighbors=5):
    """Handle missing values with mode and KNN imputation"""
    # Make copies of columns to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Categorical imputation
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
    
    # Numerical imputation
    if numerical_cols:
        cols_to_impute = [col for col in numerical_cols if col in df.columns and df[col].isna().any()]
        if cols_to_impute:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df[cols_to_impute])
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed = scaler.inverse_transform(imputer.fit_transform(scaled))
            df[cols_to_impute] = np.round(imputed, 2)
    
    return df

def clean_categorical_data(df, categorical_cols):
    """Clean and standardize categorical values"""
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace({"3+": "3", "0": "0"}).fillna("0")
    
    return df

def handle_outliers(df, outlier_cols):
    """IQR-based outlier handling"""
    df = df.copy()
    for col in outlier_cols:
        if col in df.columns and df[col].dtype in [np.int64, np.float64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            median = df[col].median()
            df[col] = np.where((df[col] < lower) | (df[col] > upper), median, df[col])
    
    return df

def feature_engineering(df):
    """Create new features"""
    df = df.copy()
    
    # Convert dependents first
    if 'Dependents' in df.columns:
        df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0).astype(int)
    
    # New features
    if all(col in df.columns for col in ["ApplicantIncome", "CoapplicantIncome"]):
        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    
    if all(col in df.columns for col in ["TotalIncome", "LoanAmount"]):
        df["Income_to_Loan_Ratio"] = df["TotalIncome"] / (df["LoanAmount"].replace(0, np.nan) + 1e-6)
    
    if 'LoanAmount' in df.columns:
        df["LoanAmount_log"] = np.log(df["LoanAmount"].replace(0, np.nan))
    
    if all(col in df.columns for col in ['Married', 'Dependents']):
        df['Family_Size'] = np.where(df['Married'] == 'Yes', 2 + df['Dependents'], 1 + df['Dependents'])
    
    if all(col in df.columns for col in ['Education', 'Self_Employed']):
        df['Is_Graduate_And_SelfEmployed'] = np.where(
            (df['Education'] == 'Graduate') & (df['Self_Employed'] == 'Yes'), 1, 0)
    
    # Rounding
    float_cols = ["TotalIncome", "Income_to_Loan_Ratio", "LoanAmount_log"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    return df

def feature_tuning(df):
    """
    Perform feature encoding and scaling for machine learning
    Returns:
        DataFrame: Processed dataframe with encoded and scaled features
        dict: Mapping of encoders used for reference
    """
    df_processed = df.copy()
    encoder_mapping = {}
    
    # 1. Handle special cases
    if 'Dependents' in df_processed.columns:
        df_processed['Dependents'] = pd.to_numeric(
            df_processed['Dependents'].replace('3+', '3').fillna('0'), 
            errors='coerce'
        ).fillna(0).astype(int)
    
    # 2. Encode categorical features
    # Binary features - Label Encoding
    binary_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']
    binary_cols = [col for col in binary_cols if col in df_processed.columns]
    
    for col in binary_cols:
        if df_processed[col].nunique() > 1:  # Only encode if more than one unique value
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoder_mapping[col] = {'encoder': 'LabelEncoder', 'classes': list(le.classes_)}
        else:
            print(f"Skipping encoding for {col} as it has only one unique value.")
    
    # Multi-category features - One-Hot Encoding
    multi_cat_cols = ['Property_Area']
    multi_cat_cols = [col for col in multi_cat_cols if col in df_processed.columns]
    
    if multi_cat_cols:
        ohe = OneHotEncoder(drop='first', sparse_output=False)
        ct = ColumnTransformer(
            [('one_hot_encoder', ohe, multi_cat_cols)],
            remainder='passthrough'
        )
        
        # Get feature names after one-hot encoding
        encoded_data = ct.fit_transform(df_processed)
        new_columns = []
        for col in ct.transformers_[0][1].get_feature_names_out(multi_cat_cols):
            new_columns.append(col.split('_')[-1])
        new_columns.extend([col for col in df_processed.columns if col not in multi_cat_cols])
        
        df_processed = pd.DataFrame(encoded_data, columns=new_columns)
        encoder_mapping['OneHotEncoder'] = {
            'columns': multi_cat_cols, 
            'categories': ct.transformers_[0][1].categories_
        }
    
    # 3. Ensuring all columns are numeric
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            try:
                df_processed[col] = pd.to_numeric(df_processed[col])
            except ValueError:
                # If conversion fails, usage of label encoding
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                encoder_mapping[col] = {'encoder': 'LabelEncoder', 'classes': list(le.classes_)}
    
    # 4. Scaling numerical features
    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    if 'Loan_Status' in numerical_cols:  
        numerical_cols.remove('Loan_Status')
    
    if numerical_cols:
        scaler = StandardScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        encoder_mapping['Scaler'] = {
            'type': 'StandardScaler', 
            'features': numerical_cols,
            'mean': scaler.mean_,
            'scale': scaler.scale_
        }
    
    return df_processed, encoder_mapping

def train_xgboost_model(df_tuned, target_col='Loan_Status', test_size=0.2, random_state=42):
    """
    Train and evaluate an XGBoost model on the processed data
    """
    # Drop non-feature columns
    non_feature_cols = ['Loan_ID']  # Adding any other columns to exclude
    feature_cols = [col for col in df_tuned.columns if col not in non_feature_cols and col != target_col]
    
    # Separate features and target
    X = df_tuned[feature_cols]
    y = df_tuned[target_col]
    
    # Rest of the function remains the same...
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=random_state,
        early_stopping_rounds=10  
    )
    
    # Basic hyperparameters
    params = {
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'min_child_weight': 1
    }
    
    xgb_model.set_params(**params)
    
    # Train model
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Making predictions
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    }
    
    # Feature importance
    importance = xgb_model.get_booster().get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    return metrics, xgb_model, dict(sorted_importance)

def plot_model_performance(metrics):
    """Visualize model performance metrics"""
    # ROC Curve
    plt.figure(figsize=(12, 5))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Confusion Matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    # Metrics bar plot
    plt.figure(figsize=(8, 4))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    metric_values = [
        metrics['accuracy'], metrics['precision'], 
        metrics['recall'], metrics['f1'], metrics['roc_auc']
    ]
    sns.barplot(x=metric_names, y=metric_values, palette='viridis')
    plt.ylim(0, 1)
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    plt.show()

def plot_feature_importance(importance_dict, max_features=15):
    """Plot feature importance"""
    features = list(importance_dict.keys())[:max_features]
    importance = list(importance_dict.values())[:max_features]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=features, palette='magma')
    plt.title('Top Feature Importance (XGBoost)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def hyperparameter_tuning(X_train, y_train, cv=5):
    """Perform hyperparameter tuning using GridSearchCV"""
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found: ", grid_search.best_params_)
    print("Best ROC AUC score: ", grid_search.best_score_)
    
    return grid_search.best_estimator_
def predict_loan_approval():
    """Collect user input and predict loan approval using the saved model"""
    try:
        # Load saved artifacts
        model = joblib.load('loan_approval_xgboost_model.pkl')
        encoders = joblib.load('encoder_mapping.pkl')
        cat_modes = joblib.load('cat_modes.pkl')
        num_medians = joblib.load('num_medians.pkl')
    except FileNotFoundError:
        print("Required model files not found. Please train the model first.")
        return

    print("\n\033[1mLoan Eligibility Checker\033[0m")
    print("Please enter the following details (press Enter to use default values):")
    
    # Collect user input with validation
    user_data = {}
    
    # Categorical features with validation
    categorical_features = {
        "Gender": ["Male", "Female"],
        "Married": ["Yes", "No"],
        "Dependents": ["0", "1", "2", "3+"],
        "Education": ["Graduate", "Not Graduate"],
        "Self_Employed": ["Yes", "No"],
        "Property_Area": ["Urban", "Semiurban", "Rural"]
    }
    
    for feature, options in categorical_features.items():
        while True:
            value = input(f"{feature} ({'/'.join(options)}): ").strip().title()
            if not value:
                value = cat_modes.get(feature, options[0])
                print(f"Using default: {value}")
                break
            if value in options:
                break
            print(f"Invalid input. Please choose from {options}")
        user_data[feature] = value

    # Numerical features with validation
    numerical_features = {
        "ApplicantIncome": (0, 100000),
        "CoapplicantIncome": (0, 100000),
        "LoanAmount": (1, 5000000),
        "Loan_Amount_Term": (12, 480),
        "Credit_History": (0, 1)
    }
    
    for feature, (min_val, max_val) in numerical_features.items():
        while True:
            value = input(f"{feature} ({min_val}-{max_val}): ").strip()
            if not value:
                value = num_medians.get(feature, min_val)
                print(f"Using default: {value}")
                break
            try:
                value = float(value)
                if min_val <= value <= max_val:
                    break
                print(f"Value must be between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")
        user_data[feature] = value

    # Create DataFrame from input
    input_df = pd.DataFrame([user_data])
    
    # Apply preprocessing pipeline
    input_df = clean_categorical_data(input_df, categorical_features.keys())
    input_df = handle_outliers(input_df, numerical_features.keys())
    input_df = feature_engineering(input_df)

    # Feature tuning transformations
    # 1. Handle special cases
    input_df['Dependents'] = input_df['Dependents'].replace('3+', '3').fillna('0').astype(int)
    
    # 2. Apply encodings
    # Binary features
    for feature in ['Gender', 'Married', 'Education', 'Self_Employed']:
        le = LabelEncoder()
        le.classes_ = np.array(encoders[feature]['classes'])
        input_df[feature] = le.transform(input_df[feature])
    
    # One-Hot Encoding
    ohe_info = encoders.get('OneHotEncoder', {})
    if ohe_info:
        for col in ohe_info['columns']:
            dummies = pd.get_dummies(input_df[col], prefix=col)
            expected_cols = [f"{col}_{cat}" for cat in ohe_info['categories'][0][1:]]
            for ec in expected_cols:
                if ec not in dummies.columns:
                    dummies[ec] = 0
            input_df = pd.concat([input_df, dummies[expected_cols]], axis=1)
            input_df.drop(col, axis=1, inplace=True)

    # 3. Scaling
    scaler_info = encoders.get('Scaler', {})
    if scaler_info:
        for feature in scaler_info['features']:
            if feature in input_df.columns:
                idx = scaler_info['features'].index(feature)
                mean = scaler_info['mean'][idx]
                std = scaler_info['scale'][idx]
                input_df[feature] = (input_df[feature] - mean) / std

    # Ensure all model features are present
    model_features = model.get_booster().feature_names
    for feature in model_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Make prediction
    prediction = model.predict(input_df[model_features])
    proba = model.predict_proba(input_df[model_features])[0][1]
    
    # Decode prediction
    status_map = {0: 'Rejected', 1: 'Approved'}
    if 'Loan_Status' in encoders:
        status_map = {i: cls for i, cls in enumerate(encoders['Loan_Status']['classes'])}
    
    print("\n\033[1mPrediction Result:\033[0m")
    print(f"Loan Status: {status_map[prediction[0]]}")
    print(f"Approval Probability: {proba:.2%}")
    print("\nRecommended Action:")
    if prediction[0] == 1:
        print("Congratulations! You're likely to get the loan approved.")
    else:
        print("Consider improving your application: Increase income, reduce debts, or improve credit history.") 
def main():
    # Configuration
    numerical_cols = [
        "ApplicantIncome", "CoapplicantIncome", 
        "LoanAmount", "Loan_Amount_Term", "Credit_History"
    ]
    categorical_cols = [
        "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "Property_Area", "Loan_Status"
    ]
    
    try:
        df = pd.read_csv("loan_data_set.csv")
        original_df = df.copy()
        print("Data loaded successfully. Shape:", df.shape)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Initial analysis
    print("\nStarting initial analysis...")
    original_stats = descriptive_analysis(original_df, "Original Data")
    
    # Processing pipeline
    print("\nProcessing data...")
    df = handle_missing_values(df, numerical_cols, categorical_cols)
    df = clean_categorical_data(df, categorical_cols)
    df = handle_outliers(df, numerical_cols[:4])
    df = feature_engineering(df)
    
    # Update numerical columns
    numerical_cols += [
        "TotalIncome", "Income_to_Loan_Ratio",
        "LoanAmount_log", "Family_Size",
        "Is_Graduate_And_SelfEmployed"
    ]
    numerical_cols = list(set(numerical_cols))  # Remove duplicates
    
    # Final analysis before feature tuning
    print("\nAnalyzing cleaned data...")
    cleaned_stats = descriptive_analysis(df, "Cleaned Data (Pre-Tuning)")
    
    # Feature tuning
    print("\nPerforming feature tuning...")
    df_tuned, encoders = feature_tuning(df)
    
    # Analysis after feature tuning
    print("\nAnalyzing final tuned data...")
    tuned_stats = descriptive_analysis(df_tuned, "Final Tuned Data")
    
    # Print encoding information
    print("\nFeature Tuning Details:")
    print("="*40)
    for feature, info in encoders.items():
        if feature == 'Scaler':
            print(f"\nNumerical Features Scaled ({info['type']}):")
            print(", ".join(info['features']))
        elif feature == 'OneHotEncoder':
            print(f"\nOne-Hot Encoded Features:")
            for col, cats in zip(info['columns'], info['categories']):
                print(f"{col}: {list(cats)}")
        else:
            print(f"\nLabel Encoded: {feature}")
            print(f"Mapping: {dict(zip(info['classes'], range(len(info['classes']))))}")
    
    # Model training and evaluation
    print("\nTraining XGBoost model...")
    metrics, model, feature_importance = train_xgboost_model(df_tuned)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print("="*40)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Visualizations
    print("\nGenerating performance visualizations...")
    plot_model_performance(metrics)
    plot_feature_importance(feature_importance)
    
    # Save the model and data
    cat_modes = df[categorical_cols].mode().iloc[0].to_dict()
    num_medians = df[numerical_cols].median().to_dict()
    
    joblib.dump(model, 'loan_approval_xgboost_model.pkl')
    joblib.dump(encoders, 'encoder_mapping.pkl')
    joblib.dump(cat_modes, 'cat_modes.pkl')
    joblib.dump(num_medians, 'num_medians.pkl')
    joblib.dump(model, 'loan_approval_xgboost_model.pkl')
    df.to_csv("cleaned_loan_data.csv", index=False)
    df_tuned.to_csv("tuned_loan_data.csv", index=False)
    
    print("\nProcessing complete. Files saved:")
    print("- cleaned_loan_data.csv (pre-tuning)")
    print("- tuned_loan_data.csv (post-tuning)")
    print("- loan_approval_xgboost_model.pkl (trained model)")

    if input("\nWould you like to check loan eligibility now? (yes/no): ").lower() == 'yes':
        predict_loan_approval()

if __name__ == "__main__":
    main()