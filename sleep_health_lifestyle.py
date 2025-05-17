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

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score

# -------------------------------------------
# Load and preview the dataset
# -------------------------------------------
df = pd.read_csv('sleep_health_and_lifestyle_dataset.csv')
df.head()

# -------------------------------------------
# Supervised Learning Models with Evaluation (Module 6)
# -------------------------------------------

# -------------------------------------------
# PRE-PROCESSING
# -------------------------------------------

# -------------------------------------------
# MODELING
# -------------------------------------------

# -------------------------------------------
# Multi-Class Logistic Regression
# Decision Tree
# Random Forest
# -------------------------------------------

# -------------------------------------------
# EVALUATION
# Cross-Comparison Between ALL Models
# -------------------------------------------

