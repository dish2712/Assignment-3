# Core Libraries
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
# Load dataset
df = pd.read_csv('your_dataset.csv')  # Replace with your dataset file

# Explore the dataset
print(df.head())  # View first few rows
print(df.info())  # Check data types and null values
print(df.describe())  # Summary statistics

# Check target class distribution
sns.countplot(x='target_column', data=df)  # Replace 'target_column' with your target variable
plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example: Resume and JD Similarity
tfidf = TfidfVectorizer()
resume_jd_matrix = tfidf.fit_transform(df['resume'] + ' ' + df['job_description'])
df['similarity_score'] = cosine_similarity(resume_jd_matrix).diagonal()
from textblob import TextBlob

df['sentiment_polarity'] = df['resume'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['transcript_length'] = df['transcript'].apply(len)
# Define X (features) and Y (target)
X = df.drop(columns=['target_column'])  # Replace 'target_column' with your target variable
y = df['target_column']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_pred))
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, y_pred_rf))
from sklearn.metrics import confusion_matrix, plot_roc_curve

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
plot_roc_curve(best_rf, X_test, y_test)
plt.title('ROC Curve')
plt.show()
importances = best_rf.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.show()