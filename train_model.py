import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
import os

def main() -> None:
    """
    Main function to load data, handle class imbalance, perform hyperparameter tuning
    via a unified pipeline without data leakage, evaluate it vigorously, 
    and save the model for FastAPI consumption.
    """
    print("Loading dataset...")
    df = pd.read_csv('Brain_Tumor_Dataset.csv')

    # Ensure compatibility: Feature names mapped explicitly
    feature_names = ['Age', 'Tumor_Size', 'Survival_Rate', 'Tumor_Growth_Rate', 'Stage', 'Tumor_Density', 'Edema_Size']
    X = df[feature_names]
    y = df['Tumor_Type']

    print("-" * 40)
    print("1. CLASS IMBALANCE CHECK")
    print("-" * 40)
    class_counts = y.value_counts()
    print(f"Class Distribution in full dataset:\n{class_counts.to_string()}")
    
    # 0 -> Benign, 1 -> Malignant. We want to weight the positive class (1) proportionally
    neg_count = class_counts.get(0, 1)
    pos_count = class_counts.get(1, 1)
    spw = neg_count / pos_count
    print(f"Calculated scale_pos_weight: {spw:.3f}")

    print("\nSplitting data into train and test sets (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("-" * 40)
    print("2 & 5. PIPELINE TRAINING AND HYPERPARAMETER TUNING")
    print("-" * 40)
    
    # Constructing a Pipeline structurally to prevent data leakage during Cross Validation
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            scale_pos_weight=spw, 
            random_state=42
        ))
    ])

    # Parameter grid for tuning
    param_distributions = {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__max_depth': [3, 5, 7, 10],
        'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'xgb__subsample': [0.6, 0.8, 1.0],
        'xgb__colsample_bytree': [0.6, 0.8, 1.0]
    }

    print("Running RandomizedSearchCV (cv=5)...")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring='f1', # Optimizing for F1 since dataset is imbalanced
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    print(f"\nBest Parameters Found: {search.best_params_}")

    print("-" * 40)
    print("EVALUATING FINAL PIPELINE ON TEST SET")
    print("-" * 40)
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    
    # Calculate precision, recall, f1 dynamically avoiding zero division constraints
    prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

    print(f"Accuracy:  {acc:.4f}\nPrecision: {prec:.4f}\nRecall:    {rec:.4f}\nF1-score:  {f1:.4f}")

    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC:   {roc_auc:.4f}")
    else:
        print("ROC-AUC:   Skipped (Only single class evaluated)")

    print("\n" + "-" * 40)
    print("3. CONFUSION MATRIX")
    print("-" * 40)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Positives (Malignant correctly identified): {tp}")
    print(f"False Positives (Benign marked as Malignant):    {fp}")
    print(f"True Negatives (Benign correctly identified):    {tn}")
    print(f"False Negatives (Malignant missed as Benign):    {fn}")
    print(f"\nMatrix Form:\n{cm}")

    print("\n" + "-" * 40)
    print("4. CLASSIFICATION REPORT")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=['0 (Benign)', '1 (Malignant)']))

    print("\n" + "-" * 40)
    print("7. PROBABILITY DISTRIBUTION (Bonus)")
    print("-" * 40)
    prob_dist = np.histogram(y_prob, bins=10, range=(0, 1))
    print("Probability Bins (0.0 to 1.0):")
    for count, edge in zip(prob_dist[0], prob_dist[1][:-1]):
        print(f"[{edge:.1f} - {edge+0.1:.1f}): {count} predictions")

    print("\n" + "-" * 40)
    print("6. PREPARING EXPORTS FOR MAIN.PY COMPATIBILITY")
    print("-" * 40)
    
    # Extract model and scaler OUT of the fitted pipeline so main.py doesn't break
    final_scaler = best_pipeline.named_steps['scaler']
    final_model = best_pipeline.named_steps['xgb']

    # Feature Importance mapped to Bar Graph securely
    importances = final_model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    print("\nTop Features by Importance:")
    print(feat_imp.to_string(index=False))

    os.makedirs('models', exist_ok=True)
    
    # Save Bar Plot natively tracking feature mappings
    plt.figure(figsize=(8, 5))
    plt.barh(feat_imp['Feature'][::-1], feat_imp['Importance'][::-1], color='#0ea5e9')
    plt.xlabel('XGBoost Relative Feature Importance')
    plt.title('Tuned Model Feature Drivers')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    
    joblib.dump(final_model, 'models/model.pkl')
    joblib.dump(final_scaler, 'models/scaler.pkl')
    print("Saved decoupled model.pkl and scaler.pkl to models/ directory successfully!")

if __name__ == '__main__':
    main()
