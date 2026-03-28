# Credit Card Fraud Detection: Random Forest vs XGBoost

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ManishBarath/Credit-Card-Fraud-Detection/blob/main/Credit_Card_Fraud_Detection_comparison_of_RF_and_XGBoost.ipynb)

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning. Given the highly imbalanced nature of fraud datasets, this project utilizes **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data. We then train and compare two powerful ensemble models: **Random Forest** and **XGBoost**.

## Dataset
The dataset used is the standard Credit Card Fraud Detection dataset. It contains transactions made by European cardholders, where the positive class (frauds) accounts for a tiny fraction of all transactions.

## Project Workflow
1. **Exploratory Data Analysis (EDA)**: Visualizing the class imbalance and feature correlations.
2. **Data Preprocessing**: Scaling the `Amount` feature and splitting the data into training and testing sets.
3. **Handling Imbalance**: Applying SMOTE to generate synthetic samples for the minority class.
4. **Model Training**: 
   * Random Forest Classifier
   * XGBoost Classifier
5. **Evaluation**: Comparing models using Classification Reports, Confusion Matrix, and ROC-AUC scores.
6. **Risk Scoring**: Generating fraud probability risk scores for individual transactions using XGBoost.

## Outputs & Visualizations

*(Note: Replace the image placeholders with the actual paths to your output images once you run the notebook and save them)*

### 1. Class Distribution & Correlation
The dataset is highly imbalanced. The correlation heatmap helps identify which features are most indicative of fraudulent activity.

### 2. Model Performance
Both models demonstrate strong predictive capabilities. The Classification Report provides essential metrics like Precision, Recall (crucial for fraud detection to minimize False Negatives), and F1-score.

### 3. Confusion Matrix (XGBoost)
The confusion matrix visualizes the True Positives (correctly identified frauds) and False Positives (legitimate transactions incorrectly flagged as fraud).
> \`![Confusion Matrix](images/confusion_matrix.png)\`

### 4. ROC Curve
The Area Under the Receiver Operating Characteristic Curve (ROC-AUC) evaluates the model's ability to distinguish between classes.
> \`![ROC Curve](images/roc_curve.png)\`

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/ManishBarath/Credit-Card-Fraud-Detection.git
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
   ```
3. Run the Jupyter Notebook `Credit_Card_Fraud_Detection_comparison_of_RF_and_XGBoost.ipynb`. Alternatively, click the "Open in Colab" badge at the top to run it directly in your browser.
