from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils.plotting import plot_train_test_accuracy, plot_learning_curve

def train_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(
        n_neighbors=7,           # więcej sąsiadów
        weights='distance',      # waga na podstawie odległości
        n_jobs=-1
    )
    # Wykres krzywej uczenia
    plot_learning_curve(model, X_train, y_train, "KNN")
    
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    plot_train_test_accuracy(train_acc, test_acc, "KNN")
    return test_acc