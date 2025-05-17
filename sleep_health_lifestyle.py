# -------------------------------------------
#   Sleep Health and Lifestyle
#   INST 414: Final Project
#   Written By: Lily Gates, Christine Lee, Sharanya Padithaya
#   University of Maryland, May 2025 
# -------------------------------------------

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold


# -------------------------------------------
# Load and preview the dataset
# -------------------------------------------
df = pd.read_csv('sleep_health_and_lifestyle_dataset.csv')
df.head()

# ===========================================
# Multi-Class Logistic Regression
# ===========================================

# -------------------------------------------
# M-C Logistic Regression: PRE-PROCESSING
# -------------------------------------------
"""
Summary:
- Copy original data to keep raw data safe.
- Clean column names to snake_case.
- Parse blood_pressure into numeric systolic and diastolic.
- Create optional binary target for binary classification experiments.
- One-hot encode categorical predictors.
- Define features (X) by dropping columns not useful for modeling.
- Keep target (y) as original strings for readability.
- Label encode target for model training.
- Print classes and sample encoded labels for verification.
"""

# Step 0: Make a copy to preserve original
df_copy = df.copy()

# Step 1: Standardize column names to snake_case
df_copy.columns = df_copy.columns.str.strip().str.lower().str.replace(' ', '_')

# Step 1.5: Fix missing or empty sleep_disorder labels (Step 2 & 3)
# Fill NaNs with 'None' and fix empty strings if any
df_copy['sleep_disorder'] = df_copy['sleep_disorder'].fillna('None')
df_copy.loc[df_copy['sleep_disorder'].str.strip() == '', 'sleep_disorder'] = 'None'

# Now check the distribution again to confirm
print("Target class distribution BEFORE encoding:")
print(df_copy['sleep_disorder'].value_counts())

# Step 2: Split blood_pressure into systolic and diastolic
df_copy[['systolic', 'diastolic']] = df_copy['blood_pressure'].str.split('/', expand=True)
df_copy['systolic'] = pd.to_numeric(df_copy['systolic'], errors='coerce')
df_copy['diastolic'] = pd.to_numeric(df_copy['diastolic'], errors='coerce')

# Step 3: Drop unused columns (e.g., person_id and original blood_pressure)
df_cleaned = df_copy.drop(columns=['person_id', 'blood_pressure'])

# Step 4: One-hot encode relevant categorical predictors
df_encoded = pd.get_dummies(
    df_cleaned,
    columns=['gender', 'occupation', 'bmi_category'],
    drop_first=True
)

# Step 5: Define features (X) and target (y) — y keeps original string labels
X = df_encoded.drop(columns=['sleep_disorder'])
y = df_encoded['sleep_disorder']  # string labels like 'None', 'Insomnia', 'Sleep Apnea'

# Step 6: Manually map target labels to desired numeric order
class_order = ['None', 'Insomnia', 'Sleep Apnea']
class_mapping = {label: idx for idx, label in enumerate(class_order)}
y_encoded = y.map(class_mapping)

# Save class order for later use in plots and reports
le_classes_ordered = np.array(class_order)

# -------------------------------------------
# M-C Logistic Regression: TRAIN & STANDARDIZE
# -------------------------------------------

# Step 1. Train-test split (stratify if classes are imbalanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Step 2. Standardize features (only needed for models like logistic regression & SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3. Initialize and train multinomial logistic regression
log_reg_model = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42
)
log_reg_model.fit(X_train_scaled, y_train)

# Step 4. Make predictions
y_pred_log_reg = log_reg_model.predict(X_test_scaled)

# -------------------------------------------
# M-C Logistic Regression: FEATURE ANALYSIS
# -------------------------------------------

# -------------------------------------------
# DataFrame of Feature Analysis
# -------------------------------------------
# Logistic Regression Coefficient-Based Feature Analysis

"""
Summary:
To interpret the influence of each predictor on classification outcomes, we analyzed the learned logistic regression coefficients for each sleep disorder class.
This coefficient-based feature analysis allows us to identify the most influential features per class.”
"""

# DataFrame of coefficients: classes as rows, features as columns
coef_df = pd.DataFrame(
    log_reg_model.coef_,           # shape: (n_classes, n_features)
    columns=X_train.columns,       # feature names
    index=le_classes_ordered       # class names in order, e.g., ['None', 'Insomnia', 'Sleep Apnea']
)

# Save coefficients to CSV file
coef_df.to_csv("logistic_regression_coefficients_by_class.csv")
print(f"\nFeature Anaysis (Logistic Regression) saved as 'logistic_regression_coefficients_by_class.csv'\n")

# Display the coefficients DataFrame
#print(coef_df)
coef_df

# -------------------------------------------
# Plotting Feature Analysis
# -------------------------------------------
for cls in coef_df.index:
    plt.figure(figsize=(10, 6))
    class_coef = coef_df.loc[cls].sort_values()

    colors = ['orange' if val < 0 else 'green' for val in class_coef.values]
    bars = plt.barh(class_coef.index, class_coef.values, color=colors)

    for bar in bars:
        width = bar.get_width()
        label = f'{width:.2f}'

        threshold = 0.1

        if abs(width) > threshold:
            xpos = width - 0.02 if width > 0 else width + 0.02
            ha = 'right' if width > 0 else 'left'
            color = 'white'
        else:
            xpos = width + 0.02 if width > 0 else width - 0.02
            ha = 'left' if width > 0 else 'right'
            color = 'black'

        plt.text(
            xpos,
            bar.get_y() + bar.get_height() / 2,
            label,
            va='center',
            ha=ha,
            fontsize=9,
            color=color,
            fontweight='bold'  # <---- added bold here
        )

    plt.title(f'Logistic Regression Coefficients for Class: {cls}', fontsize=14, pad=10, fontweight='bold')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()

    filename = f'logreg_coefficients_{cls.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot for class '{cls}' as '{filename}'")

    plt.show()

# -------------------------------------------
# M-C Logistic Regression: EVALUATTION
# -------------------------------------------

print(f"\nLogistic Regression Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}\n")
print("Logistic Regression Classification Report:\n\n", classification_report(y_test, y_pred_log_reg, target_names=le_classes_ordered))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_log_reg, labels=range(len(le_classes_ordered)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_classes_ordered)
disp.plot(cmap='Blues', xticks_rotation=45)

# Add title and layout
plt.title("Confusion Matrix: Logistic Regression")
plt.tight_layout()

# Save figure with a descriptive name
plt.savefig("confusion_matrix_logistic_regression_sleep_disorder.png", dpi=300)
print("\nConfusion Matrix (Logistic Regression) saved as 'confusion_matrix_logistic_regression_sleep_disorder.png'\n")

# Show the plot
plt.show()

# -------------------------------------------
# M-C Logistic Regression: MISCLASSIFICATIONS
# -------------------------------------------
# Create a DataFrame of test data with predictions
misclassified_df = X_test.copy()

# Map numeric labels back to original string labels using le_classes_ordered
misclassified_df['True Label'] = y_test.map(lambda x: le_classes_ordered[x])
misclassified_df['Predicted Label'] = pd.Series(y_pred_log_reg, index=y_test.index).map(lambda x: le_classes_ordered[x])

# Filter only the misclassified rows
misclassified_only = misclassified_df[misclassified_df['True Label'] != misclassified_df['Predicted Label']]

# Reset index for nicer display
misclassified_only = misclassified_only.reset_index(drop=True)

print("Number of Misclassifications:", len(misclassified_only))

# Save misclassified results to CSV
misclassified_only.to_csv("logreg_misclassified_sleep_disorder_predictions.csv", index=False)
print("\nMisclassified (log regress) data saved to 'logreg_misclassified_sleep_disorder_predictions.csv'\n")

#print(misclassified_only)
misclassified_only

# ===========================================
# Decision Tree
# ===========================================

# -------------------------------------------
# Decision Tree: PRE-PROCESSING
# -------------------------------------------

# Refine the Decision Tree Model: Find Optimal Depth

# Define a range of depths to test
depths = range(1, 21)  # You can adjust this range if needed
cv_scores = []

# StratifiedKFold is a variant of K-Fold cross-validationfold
# Helps get more reliable model performance estimates, especially when classes are imbalanced
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation for each depth
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot the results
plt.plot(depths, cv_scores, marker='o', label='Cross-Validation Accuracy')
plt.title('Cross-Validation Scores vs Tree Depth')
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validation Accuracy')

# Find the max accuracy and corresponding depth
best_depth = depths[cv_scores.index(max(cv_scores))]
best_score = max(cv_scores)

# Mark the best point
plt.scatter(best_depth, best_score, color='red', zorder=5, label=f'Best Depth: {best_depth}\nAccuracy: {best_score:.4f}')

# Add legend
plt.legend()


# Output the optimal max_depth and corresponding score
print(f"\nThe optimal max_depth is {best_depth} with a cross-validation accuracy of {best_score:.4f}\n")

# Save figure
plt.savefig("cross_val_scores_vs_depth.png", dpi=300, bbox_inches='tight')
print("\nThe cross-validation scores graph is saved as: 'cross_val_scores_vs_depth.png'\n")

# Show the plot
plt.show()

# -------------------------------------------
# Decision Tree: TRAIN
# -------------------------------------------

# Train Optimal Decision Tree
# 1. Initialize and train the Decision Tree with the best depth
dt_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_model.fit(X_train, y_train)

# 2. Predict on the test set
y_pred_dt = dt_model.predict(X_test)

# -------------------------------------------
# Graph the Decision Tree Model
# -------------------------------------------

plt.figure(figsize=(24, 10))
plot_tree(
    dt_model,
    filled=True,
    feature_names=X_train.columns,
    class_names = list(le_classes_ordered),
    rounded=True,
    fontsize=11
)

plt.title(
    f"Decision Tree for Sleep Disorder Classification\nOptimal Tree Depth = {best_depth}",
    fontsize=20,
    fontweight='bold',
    loc='center',
    pad=30,
    y=.95
)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.savefig("decision_tree_optimal_depth.png", dpi=300, bbox_inches='tight')
print("\nDecision tree graph saved as: 'decision_tree_optimal_depth.png'\n")
plt.show()

# ----------------------------------------------------------
# Get Rules for the Decision Tree Model
# ----------------------------------------------------------
# Note: Text version of tree

# Extract the decision tree rules with feature names and max depth
tree_rules = export_text(dt_model, feature_names=list(X_train.columns), max_depth=best_depth)

print(f"\nThe decision tree rules have been saved to 'decision_tree_rules.txt' ({len(tree_rules.splitlines())} lines).\n")

# Print the rules to the console (optional)
print(tree_rules)

# Save the rules to a text file
with open("decision_tree_rules.txt", "w") as f:
    f.write(tree_rules)

# -------------------------------------------
# Decision Tree: FEATURE ANALYSIS
# -------------------------------------------

# -------------------------------------------
# DataFrame for Feature Analysis
# -------------------------------------------

# Create a DataFrame for feature importances
feature_importances = dt_model.feature_importances_
feature_names = X_train.columns  # Assuming X_train is a DataFrame

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort by importance
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Define filename
feat_imp_filename = "decision_tree_feature_importances.csv"

# Save to CSV
feat_imp_df.to_csv(feat_imp_filename, index=False)

# Confirmation message
print(f"\nFeature importances saved to '{feat_imp_filename}'\n")

# Display the DataFrame
print(feat_imp_df)

# -------------------------------------------
# Plot for Feature Analysis
# -------------------------------------------

# Filter non-zero importance features and sort ascending
feat_imp_nonzero = feat_imp_df[feat_imp_df['Importance'] > 0].sort_values(by='Importance', ascending=True)

# Get tab10 colormap and create a color list for each bar
cmap = plt.get_cmap('tab10')
num_colors = len(feat_imp_nonzero)
colors = [cmap(i % cmap.N) for i in range(num_colors)]  # cycle if more than 10 bars

plt.figure(figsize=(10, 8))
bars = plt.barh(feat_imp_nonzero['Feature'], feat_imp_nonzero['Importance'], color=colors)

plt.title('Feature Importance from Decision Tree (Non-zero only)', fontweight='bold', fontsize=18)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)

# Increase tick label size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add footer note
plt.figtext(0.5, 0.01, 
            "Note: Features not shown had zero importance according to the trained Decision Tree model.",
            ha="center", fontsize=12, color='black', fontstyle='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for footer

plt.savefig("decision_tree_feature_importance.png", dpi=300, bbox_inches='tight')

print(f"\nPlot saved as 'decision_tree_feature_importance.png' in the current working directory.\n")

plt.show()

# -------------------------------------------
# Decision Tree: EVALUATION
#-------------------------------------------

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred_dt)
print(f"\nDecision Tree Accuracy: {accuracy:.4f}\n")

# Classification Report (Precision, Recall, F1-Score)
print("Decision Tree Classification Report:\n\n", classification_report(y_test, y_pred_dt, target_names=le_classes_ordered))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_dt, labels=range(len(le_classes_ordered)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_classes_ordered)
disp.plot(cmap='Blues', xticks_rotation=45)

# Add title and adjust layout
plt.title("Confusion Matrix: Decision Tree")
plt.tight_layout()

# Save figure with descriptive name
plt.savefig("confusion_matrix_decision_tree_sleep_disorder.png", dpi=300, bbox_inches='tight')
print("Confusion Matrix (Decision Tree) saved as 'confusion_matrix_decision_tree_sleep_disorder.png'\n")

# Show the plot
plt.show()


# -------------------------------------------
# Decision Tree: MISCLASSIFICATIONS
# -------------------------------------------

# Create a DataFrame of test data with predictions
misclassified_dt_df = X_test.copy()

# Map numeric labels back to original string class names using le_classes_ordered
misclassified_dt_df['True Label'] = y_test.map(lambda x: le_classes_ordered[x])
misclassified_dt_df['Predicted Label'] = pd.Series(y_pred_dt, index=y_test.index).map(lambda x: le_classes_ordered[x])

# Filter only the misclassified rows
misclassified_dt_only = misclassified_dt_df[misclassified_dt_df['True Label'] != misclassified_dt_df['Predicted Label']]

# Reset index for nicer display
misclassified_dt_only = misclassified_dt_only.reset_index(drop=True)

print("Number of Misclassifications (Decision Tree):", len(misclassified_dt_only))

# Save misclassified results to CSV
filename_dt = "decision_tree_misclassified_sleep_disorder_predictions.csv"
misclassified_dt_only.to_csv(filename_dt, index=False)
print(f"\nMisclassified (Decision Tree) data saved to '{filename_dt}'\n")

# Display the DataFrame
print(misclassified_dt_only)

# ===========================================
# Random Forest
# ===========================================
# -------------------------------------------
# Random Forest: PRE-PROCESSING
# -------------------------------------------


# -------------------------------------------
# Random Forest: TRAIN & STANDARDIZE
# -------------------------------------------

# -------------------------------------------
# Random Forest: CONFUSION MATRIX
# -------------------------------------------


# -------------------------------------------
# Random Forest: MISCLASSIFICATIONS
# -------------------------------------------


# -------------------------------------------
# EVALUATION
# Cross-Comparison Between ALL Models
# -------------------------------------------

