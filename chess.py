from utils.preprocessing import load_and_preprocess_data
from models.neural_network import train_neural_network
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from models.knn_model import train_knn

from grid_search.random_forest_gridsearch import train_random_forest_gridsearch
from grid_search.xgboost_gridsearch import train_xgboost_gridsearch
from grid_search.knn_gridsearch import train_knn_gridsearch
from grid_search.neural_network_gridsearch import train_neural_network_gridsearch

# Wczytanie i przetworzenie danych
file_path = "games.csv"
print("Wczytywanie i przetwarzanie danych...")
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, n_moves=15, for_onehot=True)

results = {}

print("Trenowanie: Neural Network...")
results["Neural Network"] = train_neural_network(X_train, X_test, y_train, y_test)
print("Trenowanie: Random Forest...") 
results["Random Forest"] = train_random_forest(X_train, X_test, y_train, y_test)
print("Trenowanie: XGBoost...")
results["XGBoost"] = train_xgboost(X_train, X_test, y_train, y_test)
print("Trenowanie: KNN...")
results["KNN"] = train_knn(X_train, X_test, y_train, y_test)

# print("Trenowanie: Neural Network (GridSearch)...")
# results["Neural Network (GridSearch)"] = train_neural_network_gridsearch(X_train, X_test, y_train, y_test)
# print("Trenowanie: Random Forest (GridSearch)...")
# results["Random Forest (GridSearch)"] = train_random_forest_gridsearch(X_train, X_test, y_train, y_test)
# print("Trenowanie: XGBoost (GridSearch)...")
# results["XGBoost (GridSearch)"] = train_xgboost_gridsearch(X_train, X_test, y_train, y_test)
# print("Trenowanie: KNN (GridSearch)...")
# results["KNN (GridSearch)"] = train_knn_gridsearch(X_train, X_test, y_train, y_test)

# Wyświetlenie wyników
print("Porównanie dokładności modeli:")
for model, accuracy in results.items():
    print(f"{model}: {accuracy:.2f}")