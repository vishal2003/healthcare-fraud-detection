# **Healthcare Claims Fraud Detection Using Machine Learning**

## **1. Introduction**
Due to the increase in the number of insurance claims in the recent past, it has become an important challenge for the healthcare organizations and insurance companies to identify the fraudulent claims. Fraudulent claims can result in significant financial losses, and detecting them manually is time-consuming and prone to errors. Therefore, the use of machine learning models to automate and enhance fraud detection has gained popularity.

In this project, we have implemented an ML solution pipeline from scratch which is used to classify an insurance claim as either fraudulent or not. We employed data from Kaggle, analyzed, cleaned, and tested several machine learning techniques to build a reliable fraud detection system. The idea is to enhance the likelihood of the insurance companies and healthcare organizations to identify fraudulent claims and thereby minimize their losses and optimize their performance.

## **2. Problem Statement**
The primary objective of this project is to develop a predictive model that can identify fraudulent claims from a dataset containing various claim and patient-related features. Specifically, we aim to answer the following questions:
- Can we accurately predict whether an insurance claim is fraudulent using machine learning techniques?
- What are the most influential factors in identifying fraudulent claims?
- How can we improve the model’s performance using hyperparameter tuning?

## **3. Data Collection**
The data used in this project was sourced from the [Kaggle Medicare Fraud Detection dataset]([https://www.kaggle.com](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)). The dataset containsfeatures related to the provider, reimbursement amounts, chronic conditions, and patient demographics. The target variable indicates whether a claim is fraudulent or not.

## **4. Methodology**

### **4.1 Data Preprocessing**
Data preprocessing is a critical step to ensure the quality and integrity of the dataset. The following steps were taken:
- **Handling Missing Values**: After merging various datasets (Inpatient claims, Outpatient claims, and Beneficiary details), missing values were handled carefully using imputation techniques and removing irrelevant rows.
- **Encoding Categorical Variables**: The categorical variables were encoded using methods like One-Hot Encoding and Label Encoding for compatibility with machine learning algorithms.
- **Feature Engineering**: We created new features, such as the count of physicians involved (`phy_count`) and the total reimbursement amount, to capture additional information that might be relevant to fraud detection.
- **Scaling and Normalization**: We applied scaling techniques to the numerical features to ensure that features with different scales do not negatively impact model performance.

### **4.2 Exploratory Data Analysis (EDA)**
In the EDA phase, we analyzed the distribution of key variables and their relationships to the target variable (fraud). This included:
- **Visualizing Data Distributions**: Histograms, box plots, and density plots were used to observe the distribution of variables such as claim amounts, provider codes, and chronic conditions.
- **Correlation Analysis**: A correlation matrix helped identify the relationships between features and the target variable.
- **Fraud Pattern Analysis**: We analyzed the patterns of fraudulent claims across demographic groups and reimbursement amounts.

### **4.3 Model Building**
We developed several machine learning models to predict fraud:
- **Logistic Regression**: A simple linear model to serve as a baseline.
- **Random Forest Classifier**: A powerful ensemble model that creates multiple decision trees to improve prediction accuracy.
- **Support Vector Machine (SVM)**: Used to find the optimal hyperplane that maximizes the margin between the fraud and non-fraud claims.
- **XGBoost**: A gradient-boosted tree-based algorithm known for its speed and performance.
- **TensorFlow Neural Network**: A custom neural network built using TensorFlow for deep learning-based fraud detection. We designed a multi-layered neural network with dense layers and ReLU activation. The model was compiled with the **Adam optimizer** and **binary cross-entropy** loss function. 

Each model was evaluated based on its ability to handle class imbalance, as the number of non-fraudulent claims far exceeded the fraudulent ones. Metrics such as accuracy, precision, recall, and F1-score were calculated for each model.

### **4.4 Hyperparameter Tuning**
To further improve the performance of our models, we implemented hyperparameter tuning using **GridSearchCV**. This allowed us to find the best combination of hyperparameters for each model. For example:
- In Random Forest, we tuned the number of trees (`n_estimators`) and the depth of trees (`max_depth`).
- For XGBoost, we tuned the learning rate (`eta`), maximum depth, and the number of boosting rounds.

After hyperparameter tuning, the best-performing model was selected for deployment.

### **4.5 Model Evaluation**
The final models were evaluated using a hold-out test dataset to ensure that they generalized well to unseen data. The evaluation metrics included:
- **Accuracy**: Proportion of correct predictions.
- **Precision**: Percentage of correctly predicted fraudulent claims out of all predicted fraudulent claims.
- **Recall**: Percentage of correctly predicted fraudulent claims out of all actual fraudulent claims.
- **F1-Score**: A harmonic mean of precision and recall to balance false positives and false negatives.

The **Gradient Boosting using GridSearchCV** model performed the best, achieving a recall score of 97%, which is crucial in fraud detection tasks where it is essential to minimize false negatives (i.e., missing fraudulent claims). The model also achieved an **accuracy of 95%**, striking a good balance between precision and recall

### **4.6 Model Deployment**
The selected Random Forest model was deployed using **Streamlit**, a Python framework that allows for rapid web application development. The deployment process involved the following steps:
1. **Saving the Model**: The best-performing model was saved as a claims_fraud_detection`.pkl` file using Joblib.
2. **Building the Streamlit Web App**: An intuitive web interface was created using Streamlit, where users can input claim details and get a prediction of whether the claim is fraudulent.
3. **Launching the Web App**: The app was deployed to the web, allowing real-time interaction with the model via the web interface.

The Streamlit app is user-friendly and allows insurance companies to input relevant claim data and receive instant fraud predictions. The interactive interface also includes easy-to-use drop-down menus and number input fields for claim details such as provider codes, reimbursement amounts, and chronic conditions.

## **5. Key Findings**
- **Feature Importance**: The most important features in predicting fraudulent claims were the total reimbursement amount, inpatient reimbursement, and the presence of chronic conditions such as Alzheimer's and heart failure.
- **Model Performance**: XGBoost, with hyperparameter tuning, provided the best trade-off between accuracy and recall, which is essential in fraud detection where missing a fraudulent claim could be costly.
- **Class Imbalance Handling**: Addressing class imbalance through methods like adjusting class weights improved the model's recall performance, ensuring fewer fraudulent claims were missed.

## **6. Challenges**
- **Data Quality**: Cleaning the data required significant preprocessing, especially handling missing values and encoding categorical variables.

## **7. Conclusion**
In this project, we successfully built and deployed a machine learning solution to detect fraudulent insurance claims. Using multiple models, hyperparameter tuning, and advanced techniques for handling class imbalance, we were able to develop a high-performing system. This solution can assist healthcare organizations and insurance companies in reducing fraud and improving claim processing efficiency.

The final solution was deployed as a **Streamlit web application**, enabling real-time interaction and predictions, which makes the model highly accessible to stakeholders and end users.

## **8. Future Work**
- **Incorporating Additional Data Sources**: Integrating more data, such as provider history or external fraud databases, could improve prediction accuracy.
- **Advanced Model Interpretability**: Using SHAP (SHapley Additive exPlanations) values to better understand the decision-making process of the model and improve trustworthiness.
- **Real-time Fraud Detection**: Implementing the model as a real-time fraud detection system to monitor live claims processing.
