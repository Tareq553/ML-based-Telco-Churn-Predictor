# Telco Churn Predictor

## Project Overview
This project addresses the challenge of predicting customer churn in the telecommunications industry. The goal is to analyze customer behavior and predict whether a customer will churn (leave the service) based on historical data. The project uses a **machine learning** pipeline to preprocess the data, handle imbalanced datasets, and train models to classify customers into "churn" and "not churn" categories. The insights gained from this project can help telecom companies take proactive measures to retain customers, improve customer satisfaction, and enhance business profitability.


## Motivation
Customer churn prediction is a critical problem for any subscription-based business. In this project, I aim to:

- Apply **machine learning techniques** to a real-world problem.
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

- **Data Preprocessing**:
   - Removed unnecessary columns (e.g., geographical information, customer ID).
   - Handled missing values and ensured data consistency.
   - Encoded categorical features using Label Encoding.
   - Balanced the dataset using SMOTE to address class imbalance.
   - Visualized key trends in customer churn.
   - Correlation analysis to identify significant features.

- **Model Building**:
  - Implemented **Logistic Regression** and **Random Forest** classifiers.
  - Trained the models using the balanced dataset.
  - Used hyperparameter tuning (**RandomizedSearchCV**) to optimize model performance.
    
- **Evaluation**:
  - Evaluated models using metrics like accuracy, precision, recall, and F1-score.
  - Identified the best-performing model based on evaluation metrics.

- **Results**:
The final model achieved the following metrics on the test dataset:

   - Accuracy: `0.82`
   - Precision: `0.84`
   - Recall: `0.81`
   - F1-Score:`0.82`
  

## Future Work
- Experiment with additional models like Gradient Boosting, XGBoost, or LightGBM.
- Explore advanced feature engineering techniques to enhance model performance.
- Integrate the model into a real-time customer management system.
- Use advanced NLP techniques for analyzing churn reasons if available in free-text format.

## License
This project is open-source and available under the **MIT** License.

## Authors

- [M Tareq Rahman](https://github.com/Tareq553)
