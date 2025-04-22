import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

df = pd.read_csv('data/heart.csv')
non_numeric = df.select_dtypes(include=['object']).columns.tolist()
if 'date' in non_numeric:
    df = df.drop(columns=['date'])
if df['target'].isnull().any():
    print('Warning: Found NaN in target. Dropping such rows.')
    df = df.dropna(subset=['target'])
if df.isnull().any().any():
    print('Warning: Found NaN in features. Dropping such rows.')
    df = df.dropna()
X = df.drop('target', axis=1)
y = df['target']
categorical_cols = [
    'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ),
    'SVM': SVC(
        probability=True,
        random_state=42
    ),
    'KNN': __import__('sklearn.neighbors').neighbors.KNeighborsClassifier(),
    'GradientBoosting': __import__('sklearn.ensemble').ensemble.GradientBoostingClassifier(
        random_state=42
    ),
    'DecisionTree': __import__('sklearn.tree').tree.DecisionTreeClassifier(random_state=42),
    'ExtraTrees': __import__('sklearn.ensemble').ensemble.ExtraTreesClassifier(random_state=42)
}

try:
    import lightgbm as lgb
    models['LightGBM'] = lgb.LGBMClassifier(random_state=42)
except ImportError:
    pass


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}
best_score = 0
best_model = None
best_model_name = ''

for name, model in models.items():
    try:
        clf = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, 'predict_proba'):
            y_proba = clf.predict_proba(X_test)[:, 1]
        else:
            y_proba = clf.decision_function(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc
        }
        if auc > best_score:
            best_score = auc
            best_model = clf
            best_model_name = name
    except Exception as e:
        print(f"Model {name} failed: {e}")

print('Model evaluation results:')
for name, metrics in results.items():
    print(f"{name}: {metrics}")
print(f"\nBest model: {best_model_name} (AUC = {best_score:.4f})")

os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print('Best model saved to models/model.pkl')

metrics_df = pd.DataFrame(results).T
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
for idx, metric in enumerate(metrics):
    ax = axes[idx//3, idx % 3]
    sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=ax)
    ax.set_title(f'Model Comparison: {metric}')
    ax.set_ylabel(metric.capitalize())
    ax.set_xlabel('Model')
    ax.set_xticklabels(metrics_df.index, rotation=30)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('models/model_comparison.png')
print('Graphical report saved to models/model_comparison.png')

try:
    import plotly.express as px
    import plotly.io as pio
    html_path = 'models/model_comparison.html'
    for metric in metrics:
        fig = px.bar(
            metrics_df,
            x=metrics_df.index,
            y=metric,
            title=f'Model Comparison: {metric}'
        )
        pio.write_html(
            fig,
            file=html_path,
            auto_open=False,
            include_plotlyjs='cdn'
        )
    print(f'Interactive HTML report saved to {html_path}')
except ImportError:
    print('plotly not installed, skipping HTML report.')
