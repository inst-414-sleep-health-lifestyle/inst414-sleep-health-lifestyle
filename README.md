# Sleep, Health, and Lifestyle Prediction

*Written By: Lily Gates, Christine Lee, Sharanya Padithaya*  
*University of Maryland, 2025*

## Description
This project explores how sleep patterns, health indicators, and lifestyle behaviors contribute to overall well-being by predicting categorical health outcomes using machine learning. We employ three supervised learning models—Logistic Regression, Decision Tree, and Random Forest—to classify outcomes based on input features related to sleep, physical activity, and demographic variables.

The goal is to identify which features are most predictive of health status and assess model performance across multiple evaluation metrics.

## Methodology
The analysis uses supervised learning, employing three classification models: Logistic Regression, Decision Trees, and Random Forest. 

The methodology includes:
* **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
* **Model Training**: Train Logistic Regression, Decision Tree, and Random Forest models. Use an 80/20 train-test split to evaluate generalizability
* **Model Evaluation**: The models are evaluated using performance metrics like accuracy, precision, recall, and F1-score. A confusion matrix is also used to assess model performance.
* **Feature Importance**: Analyze top predictors based on feature importance scores from tree-based models


## Required Dependencies
To run the project, the following Python libraries are required:
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`

## Output
The script generates:
* **Feature Importance Plots**: Visualizations showing the most influential factors in predicting survival (e.g., age, gender, fare).
* **Confusion Matrix**: For each model, visualizing true positives, false positives, true negatives, and false negatives.
* **Model Performance Metrics**: Including accuracy, precision, recall, and F1-score for each model.

## Limitations
* The models are based on a specific dataset and may not generalize to other populations or health domains
* Correlation does not imply causation; predictive power does not confirm clinical significance
* Imbalanced classes or noisy labels may impact performance metrics

## Future Improvements
* Hyperparameter tuning and cross-validation to optimize model performance.
