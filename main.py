from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import numpy as np
import pandas as pd
import time

df = pd.read_csv('dataset_phishing.csv')
X = df.drop(['url', 'status'], axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base_model = RandomForestClassifier(n_estimators=100, random_state=42)
base_model.fit(X_train, y_train)

y_pred = base_model.predict(X_test)
print(classification_report(y_test, y_pred))


# Eksperyment 1
n_trees_list = [5, 10, 25, 50, 100, 200, 500]
wyniki = []

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

for n in n_trees_list:
    exp_model = RandomForestClassifier(n_estimators=n, random_state=42)
    
    start = time.time()
    scores = cross_val_score(exp_model, X, y, cv=rskf, scoring='accuracy')
    czas = time.time() - start
    
    wyniki.append({
        'n_drzew': n,
        'accuracy_mean': scores.mean(),
        'accuracy_std': scores.std(),
        'czas_s': czas
    })

wyniki_df = pd.DataFrame(wyniki)
print(wyniki_df)