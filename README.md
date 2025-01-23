# Telco Churn Predictor

## Project Overview
This is a **Machine learning-based project** focused on addressing the challenge of customer churn prediction in the telecommunications industry. The objective is to analyze customer behavior and predict whether a customer is likely to churn (discontinue the service) based on historical data. The project involves building a robust machine learning pipeline to preprocess data, handle imbalanced datasets, and train classification models to categorize customers into "churn" and "not churn." Insights derived from this project will enable telecom companies to implement proactive retention strategies, enhance customer satisfaction, and drive business profitability.


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
