from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os

def train_random_forest_gridsearch(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=285733, n_jobs=-1),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    os.makedirs('best_params', exist_ok=True)
    with open('best_params/random_forest_gridsearch.txt', 'w') as f:
        f.write('\nBest params: ' + str(grid.best_params_))
        f.write('\nBest score: ' + str(grid.best_score_))
        best_model = grid.best_estimator_
        test_acc = accuracy_score(y_test, best_model.predict(X_test))
        f.write('\nTest accuracy: ' + str(test_acc))
    return test_acc