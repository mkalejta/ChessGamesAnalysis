from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils.plotting import plot_train_test_accuracy, plot_learning_curve

def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=500,        # więcej drzew
        max_depth=20,            # ograniczenie głębokości drzewa
        min_samples_split=2,     # minimalna liczba próbek do podziału węzła
        min_samples_leaf=2,      # minimalna liczba próbek w liściu
        class_weight='balanced', # balansowanie klas
        random_state=285733,
        n_jobs=-1                # używaj wszystkich rdzeni CPU
    )
    # Wykres krzywej uczenia
    plot_learning_curve(model, X_train, y_train, "Random Forest")
    
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    plot_train_test_accuracy(train_acc, test_acc, "Random Forest")
    return test_acc