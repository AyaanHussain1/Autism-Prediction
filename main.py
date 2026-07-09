import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from xgboost import XGBClassifier
import pickle

df =  pd.read_csv("final_data.xls")
# print(df)

x = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2,stratify=y)

# Smote for resampling
x_train_smote,y_train_smote = SMOTE().fit_resample(x_train,y_train) 

# Model Training 
models = {"Decision Tree" : DecisionTreeClassifier(),"random forest":RandomForestClassifier(),
         "xgboost" : XGBClassifier(random_state=42)}

# cross validation Results
cv_scores = {}

for name, model in models.items():
   
    print(f"Training {name} with default parameters")
    scores = cross_val_score(model, x_train_smote, y_train_smote, cv=5, scoring="accuracy")
    cv_scores[name] = scores
    print(f"{name} Cross Validation Score {np.mean(scores):.2f}")
    print("-" * 50)

# print(cv_scores)

# Increasing performance of models with different parameters

decision_tree = DecisionTreeClassifier(random_state=42)
randome_forest = RandomForestClassifier(random_state=42)
xgboost = XGBClassifier(random_state=42)

param_grid_dt = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20, 30, 40, 50, 60],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

param_grid_rf = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [None, 3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False] 
}

param_grid_xgb = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "subsample": [0.5, 0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0]
}

# Hyper tuning for tree based models
# performing Randomized Serach CV

randome_search_dt = RandomizedSearchCV(estimator=decision_tree,param_distributions=param_grid_dt,n_iter=10,cv=5,scoring="accuracy",random_state=42)
random_search_rf = RandomizedSearchCV(estimator=randome_forest,param_distributions=param_grid_rf,n_iter=10,cv=5,scoring="accuracy",random_state=42)
random_search_xgb = RandomizedSearchCV(estimator=xgboost,param_distributions=param_grid_xgb,n_iter=10,cv=5,scoring="accuracy",random_state=42)

randome_search_dt.fit(x_train_smote,y_train_smote)
random_search_rf.fit(x_train_smote,y_train_smote)
random_search_xgb.fit(x_train_smote,y_train_smote)

best_model = None
best_score = 0
if random_search_rf.best_score_ > best_score:
    best_model = random_search_rf.best_estimator_
    best_score  = random_search_rf.best_score_

if randome_search_dt.best_score_ > best_score:
    best_model = randome_search_dt.best_estimator_
    best_score  = randome_search_dt.best_score_

if random_search_xgb.best_score_ > best_score:
    best_model = random_search_xgb.best_estimator_
    best_score  = random_search_xgb.best_score_

# print(f"Best Model {best_model}")
# print(f"Best cross validation accuracy {best_score:.2f}")

with open("best_model.pkl","wb") as f:
    pickle.dump(best_model,f)

y_test_pred = best_model.predict(x_test)
print(f"Accuracy Score : {accuracy_score(y_test_pred,y_test)}")
print(f"confusion Matrix : {confusion_matrix(y_test_pred,y_test)}")
print(f"Classification Report : {classification_report(y_test_pred,y_test)}")