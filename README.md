# Customer-Churn-Prediction-System
A machine learning project that predicts whether a customer is likely to leave a service (churn) based on customer demographic and account information. This system analyzes customer behavior patterns and helps businesses identify customers who are at risk of leaving.

Project Overview

Customer churn prediction is an important application of data science in industries such as banking, telecom, and subscription services. By identifying customers likely to leave, companies can take preventive actions to improve retention and customer satisfaction.

This project uses machine learning classification algorithms to analyze customer data and predict churn probability.

The workflow includes:

Data preprocessing and cleaning
Exploratory Data Analysis (EDA)
Feature engineering
Training multiple machine learning models
Model evaluation and comparison
Saving the best performing model for prediction

Features

Exploratory Data Analysis to understand customer behavior
Data preprocessing and feature engineering
Implementation of multiple machine learning algorithms
Model evaluation using accuracy, precision, recall, F1-score and ROC-AUC
Cross-validation for reliable model performance
Saved trained model using Pickle for future predictions

Technologies Used

Python
Pandas
NumPy
Matplotlib
Scikit-learn
Jupyter Notebook

Machine Learning Models Used

The following models were implemented and compared:
Logistic Regression
Random Forest
Gradient Boosting
AdaBoost
Support Vector Machine (SVM)

The best model was selected based on evaluation metrics and cross-validation results.

Project Structure
customer-churn-prediction
│
├── Analysis.ipynb              # Data analysis and model training notebook
├── customer_data.csv           # Dataset used for training
├── best_churn_pipeline.pkl     # Saved trained ML pipeline
├── README.md                   # Project documentation
└── requirements.txt            # Required Python libraries
