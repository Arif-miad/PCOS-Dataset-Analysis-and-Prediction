 

---

# PCOS Dataset Analysis and Prediction  

This repository contains an in-depth analysis and machine learning implementation for the **Polycystic Ovary Syndrome (PCOS) Dataset**. The dataset aims to facilitate research on the impact of lifestyle choices on PCOS prevalence. The analysis and implementation are designed for exploratory data analysis, predictive modeling, and healthcare research.  

## Dataset Description  

The primary goal of the dataset is to enable:  
- **Exploratory Data Analysis (EDA)**: To uncover patterns, trends, and correlations in health and lifestyle metrics.  
- **Predictive Modeling**: To develop machine learning models for predicting PCOS outcomes.  
- **Health Research**: To support studies on how lifestyle choices impact reproductive health.  
- **Awareness**: To educate individuals and healthcare providers about PCOS management.  

The dataset serves as a valuable resource for data scientists, researchers, and students exploring real-world health data.  

## Project Features  

### 1. Data Preprocessing and Analysis  
- Import necessary libraries.  
- Load the dataset.  
- Explore the dataset for initial insights.  
- Handle missing and duplicate values.  
- Perform exploratory data analysis (EDA) to identify trends and patterns.  

### 2. Data Preparation  
- **Feature Engineering**: Identify and create relevant features to improve model performance.  
- **Encoding**: Convert categorical features into numerical representations using techniques like one-hot encoding or label encoding.  
- **Scaling**: Normalize features to ensure uniformity in machine learning models.  
- **Target Variable**: Analyze and prepare the target variable for classification.  

### 3. Splitting the Dataset  
- Split the dataset into training and testing sets to evaluate model performance.  

### 4. Machine Learning Models  
Define, train, and evaluate 10 classification models to predict PCOS outcomes:  
1. Logistic Regression  
2. Decision Tree  
3. Random Forest  
4. Gradient Boosting  
5. XGBoost  
6. Support Vector Machine (SVM)  
7. Naive Bayes  
8. K-Nearest Neighbors (KNN)  
9. Neural Network (MLPClassifier)  
10. AdaBoost  

### 5. Performance Metrics  
Evaluate model performance using:  
- Accuracy  
- Precision, Recall, and F1-score (via Classification Report)  
- Confusion Matrix  
- **ROC and AUC Curves**  

## Code Implementation Steps  

1. **Import Libraries**  
   Load essential libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.  

2. **Load Dataset**  
   Read the dataset and display basic information like shape, column names, and data types.  

3. **Explore Dataset**  
   - Check for missing and duplicate values.  
   - Summarize data statistics.  

4. **Handle Missing Values**  
   Fill missing values using the mean, median, or mode as appropriate.  

5. **Handle Duplicate Values**  
   Remove duplicate rows to ensure data integrity.  

6. **Exploratory Data Analysis (EDA)**  
   - Visualize distributions of features.  
   - Analyze relationships using correlation matrices and pair plots.  

7. **Feature Engineering**  
   Create and select relevant features for better predictions.  

8. **Encoding and Scaling**  
   - Encode categorical features using one-hot or label encoding.  
   - Scale numerical features for uniformity.  

9. **Target Variable Analysis**  
   Analyze and prepare the target variable for classification.  

10. **Split Dataset**  
    Split the dataset into training and testing subsets.  

11. **Model Implementation**  
    Train 10 classification models and evaluate their performance.  

12. **Performance Metrics**  
    Generate evaluation metrics like accuracy, classification reports, confusion matrices, and ROC-AUC curves for all models.  

## How to Use  

1. Clone this repository:  
   ```bash  
   git clone https://github.com/your-username/pcos-dataset-analysis.git  
   ```  

2. Install required Python libraries:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Run the Jupyter Notebook or Python script to execute the analysis:  
   ```bash  
   jupyter notebook pcos_analysis.ipynb  
   ```  

## Kaggle Notebook  

The full implementation is also available on Kaggle for interactive exploration:  
[Link to Kaggle Notebook](#) *((https://www.kaggle.com/code/arifmia/classification-with-machine-learning-models))*  

## Contributions  



## Step-by-Step Code Implementation

### 1. **Import Libraries**
```python
# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
```

### 2. **Load Dataset**
```python
# Load the dataset
dataset = pd.read_csv('path_to_your_dataset.csv')

# Display the first few rows of the dataset
dataset.head()

# Check for basic info like nulls and data types
dataset.info()

# Check the shape of the dataset
dataset.shape
```

### 3. **Explore Dataset**
```python
# Checking for missing values
dataset.isnull().sum()

# Checking for duplicate values
dataset.duplicated().sum()

# Descriptive statistics
dataset.describe()

# Visualize the distribution of numerical features
dataset.hist(figsize=(10,10))
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.show()
```

### 4. **Handle Missing Values**
```python
# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Choose the appropriate strategy (mean/median)
dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset))

# Check if there are still missing values
dataset_imputed.isnull().sum()
```

### 5. **Handle Duplicate Values**
```python
# Remove duplicate rows
dataset_clean = dataset.drop_duplicates()

# Verify if any duplicates remain
dataset_clean.duplicated().sum()
```

### 6. **Exploratory Data Analysis (EDA)**
```python
# Plot pairplot to check relationships between features
sns.pairplot(dataset_clean, hue='target_variable')  # Replace 'target_variable' with your target column
plt.show()

# Countplot to visualize target variable distribution
sns.countplot(x='target_variable', data=dataset_clean)
plt.show()
```

### 7. **Feature Engineering**
```python
# Example: Create new features based on domain knowledge or existing data
dataset_clean['new_feature'] = dataset_clean['feature1'] * dataset_clean['feature2']

# Check new features
dataset_clean.head()
```

### 8. **Encoding and Scaling**
```python
# Encoding categorical features if any
label_encoder = LabelEncoder()
dataset_clean['encoded_column'] = label_encoder.fit_transform(dataset_clean['categorical_column'])

# Standardize numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(dataset_clean.drop(['target_variable'], axis=1))  # Excluding target column

# Check the scaled features
scaled_features[:5]
```

### 9. **Target Variable Analysis**
```python
# Separate the target variable from the features
X = dataset_clean.drop('target_variable', axis=1)  # Features
y = dataset_clean['target_variable']  # Target variable
```

### 10. **Split Dataset into Training and Testing**
```python
# Split the dataset into training and testing sets (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of train and test sets
X_train.shape, X_test.shape
```

### 11. **Model Implementation**
```python
# Define a dictionary to hold models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 'N/A'
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report: \n{report}")
    print(f"ROC AUC Score: {roc_auc}")
    
    # Plot ROC curve
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# Show ROC curve
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()
```

### 12. **Performance Metrics**
```python
# Example: Plot ROC AUC curve for one model
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
```

---

### Notes:
- **Dataset Path**: Ensure the path to the dataset (`path_to_your_dataset.csv`) is correct in the code.
- **Target Variable**: Replace `target_variable` with the actual column name for the target variable in your dataset.
- **Models**: This implementation uses 10 different classification models. You can modify the models and hyperparameters as per your requirement.
- **Metric Evaluation**: The code includes accuracy, classification report, confusion matrix, and ROC-AUC curves for evaluation.



