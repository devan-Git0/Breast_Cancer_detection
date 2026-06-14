"""
Breast Cancer Detection & Classification

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Load the breast cancer dataset
print("=" * 60)
print("BREAST CANCER DETECTION & CLASSIFICATION")
print("=" * 60)

data = load_breast_cancer()
X = data.data
y = data.target

print(f"\nDataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes: {np.unique(y)} (0: Malignant, 1: Benign)")
print(f"Class distribution:\n  Malignant: {sum(y==0)}\n  Benign: {sum(y==1)}")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain-Test Split:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Define K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 60)
print("MODEL EVALUATION WITH K-FOLD CROSS-VALIDATION (K=5)")
print("=" * 60)

# Models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

cv_results = {}

# Train and evaluate each model with K-fold cross-validation
for model_name, model in models.items():
    print(f"\n{model_name}:")
    print("-" * 40)
    
    # K-fold cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"Std Dev: {cv_scores.std():.4f}")
    print(f"Min: {cv_scores.min():.4f}, Max: {cv_scores.max():.4f}")
    
    cv_results[model_name] = cv_scores
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"\nTest Set Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

# Select the best model (Random Forest typically performs best)
print("\n" + "=" * 60)
print("FINAL MODEL: RANDOM FOREST")
print("=" * 60)

best_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': data.feature_names,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# 2. K-Fold Cross-Validation Scores
cv_data = pd.DataFrame(cv_results)
cv_data.boxplot(ax=axes[0, 1])
axes[0, 1].set_title('K-Fold Cross-Validation Scores (K=5)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].grid(True, alpha=0.3)

# 3. Top 10 Feature Importance
top_features = feature_importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['importance'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
axes[1, 0].invert_yaxis()

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/breast_cancer_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'breast_cancer_results.png'")

plt.show()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✓ Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"✓ ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"✓ K-Fold CV (Mean): {cv_results['Random Forest'].mean():.4f}")
print("=" * 60)
