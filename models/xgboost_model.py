from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from utils.plotting import plot_train_test_accuracy, plot_learning_curve

def train_xgboost(X_train, X_test, y_train, y_test):
    model = XGBClassifier(
        n_estimators=500,        # więcej drzew
        max_depth=8,             # większa głębokość
        learning_rate=0.1,       # mniejsze tempo uczenia
        subsample=1.0,           # losowy podzbiór próbek
        colsample_bytree=0.7,    # losowy podzbiór cech
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=285733,
        n_jobs=-1                # używaj wszystkich rdzeni CPU
    )
    # Wykres krzywej uczenia
    plot_learning_curve(model, X_train, y_train, "XGBoost")
    
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    plot_train_test_accuracy(train_acc, test_acc, "XGBoost")
    return test_acc