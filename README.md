# Telco Churn Predictor

## Project Overview
This project addresses the challenge of predicting customer churn in the telecommunications industry. The goal is to analyze customer behavior and predict whether a customer will churn (leave the service) based on historical data. The project uses a machine learning pipeline to preprocess the data, handle imbalanced datasets, and train models to classify customers into "churn" and "not churn" categories. The insights gained from this project can help telecom companies take proactive measures to retain customers, improve customer satisfaction, and enhance business profitability.


## Motivation
Customer churn prediction is a critical problem for any subscription-based business. In this project, I aim to:

- Apply machine learning techniques to a real-world problem.
- Develop skills in handling imbalanced datasets and model evaluation. This project forms a foundation for tackling other business-critical machine learning problems and implementing data-driven strategies.


## Used Technologies
The following tools and libraries were used in this project:

- Python: Programming language for implementation.
- Pandas and NumPy: For data manipulation and analysis.
- Matplotlib and Seaborn: For data visualization and EDA (Exploratory Data Analysis).
- Scikit-learn: For preprocessing, building machine learning models, and evaluating their performance.
- Imbalanced-learn (SMOTE): To handle class imbalance by oversampling the minority class.
- Jupyter Notebook: For an interactive and organized development environment.

## Dataset Description
A fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3. The data is collected from [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset).

## Project Workflow
Data Preprocessing:


## 🔧 Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Pandas, NumPy: Data manipulation and analysis
  - Scikit-learn: Model training and evaluation
  - Seaborn, Matplotlib: Data visualization
  - Imbalanced-learn: Handling imbalanced datasets with SMOTE

Removed unnecessary columns (e.g., geographical information, customer ID).
Handled missing values and ensured data consistency.
Encoded categorical features using Label Encoding.
Balanced the dataset using SMOTE to address class imbalance.
Exploratory Data Analysis (EDA):

Visualized key trends in customer churn.
Correlation analysis to identify significant features.
Model Building:

Implemented Logistic Regression and Random Forest classifiers.
Trained the models using the balanced dataset.
Used hyperparameter tuning (RandomizedSearchCV) to optimize model performance.
Evaluation:

Evaluated models using metrics like accuracy, precision, recall, and F1-score.
Identified the best-performing model based on evaluation metrics.

Results
The final model achieved the following metrics on the test dataset:

Accuracy: [Insert Accuracy Score]
Precision: [Insert Precision Score]
Recall: [Insert Recall Score]
F1-Score: [Insert F1-Score]

Future Work
Experiment with additional models like Gradient Boosting, XGBoost, or LightGBM.
Explore advanced feature engineering techniques to enhance model performance.
Integrate the model into a real-time customer management system.
Use advanced NLP techniques for analyzing churn reasons if available in free-text format.

License
This project is open-source and available under the MIT License.

Authors
M Tareq Rahman
