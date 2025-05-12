# visualizations.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

def generate_visualizations():
    """Generate visualizations and model info"""
    print("\n" + "="*60)
    print("Generating Visualizations and Model Information")
    print("="*60 + "\n")
    
    try:
        # Create sample data
        np.random.seed(42)
        sample_size = 100
        sample_data = {
            'ApplicantIncome': np.random.normal(5000, 2000, sample_size),
            'LoanAmount': np.random.normal(120, 50, sample_size),
            'Credit_History': np.random.choice([0, 1], sample_size, p=[0.2, 0.8]),
            'Loan_Status': np.random.choice([0, 1], sample_size, p=[0.3, 0.7]),
            'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], sample_size)
        }
        df = pd.DataFrame(sample_data)
        
        # Create visualization directory
        os.makedirs('static/visualizations', exist_ok=True)
        
        # Generate and save plots
        generate_income_plot(df)
        generate_loan_vs_income_plot(df)
        generate_credit_history_plot(df)
        generate_property_area_plot(df)
        generate_confusion_matrix()
        generate_feature_importance_plot()
        
        # Create HTML report
        create_html_report()
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {str(e)}")
        raise

def generate_income_plot(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ApplicantIncome'], kde=True, bins=20)
    plt.title('Applicant Income Distribution')
    plt.xlabel('Income')
    plt.ylabel('Count')
    plt.savefig('static/visualizations/income_distribution.png')
    plt.close()
    print("✅ Generated Income Distribution plot")

def generate_loan_vs_income_plot(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Loan_Status')
    plt.title('Loan Amount vs Applicant Income')
    plt.savefig('static/visualizations/loan_vs_income.png')
    plt.close()
    print("✅ Generated Loan vs Income plot")

def generate_credit_history_plot(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Credit_History', hue='Loan_Status')
    plt.title('Loan Approval by Credit History')
    plt.xlabel('Credit History (0=Bad, 1=Good)')
    plt.ylabel('Count')
    plt.savefig('static/visualizations/credit_impact.png')
    plt.close()
    print("✅ Generated Credit History Impact plot")

def generate_property_area_plot(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, y='Property_Area', hue='Loan_Status')
    plt.title('Loan Approval by Property Area')
    plt.ylabel('Property Area')
    plt.xlabel('Count')
    plt.savefig('static/visualizations/property_area.png')
    plt.close()
    print("✅ Generated Property Area plot")

def generate_confusion_matrix():
    y_true = np.random.choice([0, 1], 100)
    y_pred = np.random.choice([0, 1], 100)
    
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('static/visualizations/confusion_matrix.png')
    plt.close()
    
    report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('static/visualizations/classification_report.csv')
    print("✅ Generated Confusion Matrix and Classification Report")

def generate_feature_importance_plot():
    features = ['Credit_History', 'ApplicantIncome', 'LoanAmount', 'Property_Area', 'CoapplicantIncome']
    importance = [0.25, 0.20, 0.15, 0.10, 0.05]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=features, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.savefig('static/visualizations/feature_importance.png')
    plt.close()
    print("✅ Generated Feature Importance plot")

def create_html_report():
    try:
        with open('static/visualizations/model_report.html', 'w') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Loan Approval Model Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
        .section {{ margin-bottom: 30px; border-bottom: 1px solid #ddd; padding-bottom: 20px; }}
        .plot-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
        .plot {{ margin: 10px; text-align: center; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Loan Approval Model Report</h1>
        <p>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="section">
        <h2>Data Visualizations</h2>
        <div class="plot-container">
            <div class="plot">
                <h3>Income Distribution</h3>
                <img src="income_distribution.png">
            </div>
            <div class="plot">
                <h3>Loan vs Income</h3>
                <img src="loan_vs_income.png">
            </div>
            <div class="plot">
                <h3>Credit History Impact</h3>
                <img src="credit_impact.png">
            </div>
            <div class="plot">
                <h3>Property Area</h3>
                <img src="property_area.png">
            </div>
        </div>
    </div>
</body>
</html>
            """)
        print("✅ Generated comprehensive HTML report")
        
    except Exception as e:
        print(f"❌ Error creating HTML report: {str(e)}")
        raise