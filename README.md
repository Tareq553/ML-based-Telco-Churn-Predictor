# Telco Churn Predictor

## Project Overview
This project addresses the challenge of predicting customer churn in the telecommunications industry. The goal is to analyze customer behavior and predict whether a customer will churn (leave the service) based on historical data. The project uses a machine learning pipeline to preprocess the data, handle imbalanced datasets, and train models to classify customers into "churn" and "not churn" categories. The insights gained from this project can help telecom companies take proactive measures to retain customers, improve customer satisfaction, and enhance business profitability.


## Motivation
Customer churn prediction is a critical problem for any subscription-based business. In this project, I aim to:

- Apply machine learning techniques to a real-world problem.
- Develop skills in handling imbalanced datasets and model evaluation. This project forms a foundation for tackling other business-critical machine learning problems and implementing data-driven strategies.


## Technologies Used
- **Programming Language**: Python
- **Platform**: Jupyter Notebook
- **Libraries**:
  - Pandas, NumPy: Data manipulation and analysis
  - Scikit-learn: Model training and evaluation
  - Seaborn, Matplotlib: Data visualization
  - Imbalanced-learn: Handling imbalanced datasets with SMOTE
    

## Dataset Details
- **Source**: [Telco customer churn: IBM dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
- **Size**: 7,043 rows, 33 columns
- **Features**:
  - Demographic Information: `gender`, `SeniorCitizen`, `Partner`, etc.
  - Subscription Details: `Contract`, `PaymentMethod`, `MonthlyCharges`, etc.
  - Target Variable: `Churn`

## Project Workflow
 

                                          +---------------------------+
                                          |    **Data Preprocessing**      |
                                          +---------------------------+
                                          | 1. Remove Unnecessary      |
                                          |    Columns                 |
                                          | 2. Handle Missing Values   |
                                          | 3. Encode Categorical      |
                                          |    Features                |
                                          | 4. Balance Dataset (SMOTE) |
                                          | 5. Visualize Key Trends    |
                                          | 6. Correlation Analysis    |
                                          +---------------------------+
                                                      |
                                                      v
                                          +---------------------------+
                                          |    **Model Building**          |
                                          +---------------------------+
                                          | 1. Logistic Regression     |
                                          | 2. Random Forest Classifier|
                                          | 3. Train Models            |
                                          | 4. Hyperparameter Tuning   |
                                          |    (RandomizedSearchCV)    |
                                          +---------------------------+
                                                      |
                                                      v
                                          +---------------------------+
                                          |    **Evaluation**              |
                                          +---------------------------+
                                          | 1. Evaluate with Accuracy, |
                                          |    Precision, Recall, F1-  |
                                          |    Score                   |
                                          | 2. Identify Best Performing|
                                          |    Model                   |
                                          +---------------------------+
                                                      |
                                                      v
                                          +---------------------------+
                                          |         **Results**            |
                                          +---------------------------+
                                          | - Accuracy: 0.82           |
                                          | - Precision: 0.84          |
                                          | - Recall: 0.81             |
                                          | - F1-Score: 0.82           |
                                          +---------------------------+



  

## Future Work
- Experiment with additional models like Gradient Boosting, XGBoost, or LightGBM.
- Explore advanced feature engineering techniques to enhance model performance.
- Integrate the model into a real-time customer management system.
- Use advanced NLP techniques for analyzing churn reasons if available in free-text format.

## License
This project is open-source and available under the **MIT** License.

## Authors

- [M Tareq Rahman](https://github.com/Tareq553)
