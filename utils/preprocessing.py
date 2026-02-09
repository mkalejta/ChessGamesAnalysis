import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import numpy as np

def load_and_preprocess_data(file_path, n_moves=15, for_nn=False, for_onehot=False):
    # Wczytanie danych z pliku CSV
    data = pd.read_csv(file_path)
    
    # Rozdzielenie ruchów na listę
    data['moves_list'] = data['moves'].str.split(' ')
    
    # Usunięcie partii zbyt krótkich
    data = data[data['moves_list'].apply(lambda x: len(x) >= 2 * n_moves)].copy()
    
    # Ekstrakcja pierwszych 2*n ruchów jako osobne kolumny
    for i in range(2 * n_moves):
        data[f'move_{i+1}'] = data['moves_list'].apply(lambda x: x[i])
    
    # Wyodrębnienie godziny rozpoczęcia partii
    data['created_hour'] = pd.to_datetime(data['created_at'], unit='ms').dt.hour
    
    # Wyodrębnienie dnia tygodnia rozpoczęcia partii
    data['created_weekday'] = pd.to_datetime(data['created_at'], unit='ms').dt.weekday
    
    # Obliczenie czasu trwania partii w sekundach
    data['game_duration'] = (data['last_move_at'] - data['created_at']) // 1000

    # Utworzenie zbioru cech X i etykiet y
    X = data[[f'move_{i+1}' for i in range(2 * n_moves)]].copy()
    X['opening_eco'] = data['opening_eco']
    X['opening_name'] = data['opening_name']
    X['opening_ply'] = data['opening_ply']
    X['turns'] = data['turns']
    X['rated'] = data['rated'].astype(int)
    X['increment_code'] = data['increment_code']
    X['victory_status'] = data['victory_status']
    X['created_hour'] = data['created_hour']
    X['created_weekday'] = data['created_weekday']
    X['game_duration'] = data['game_duration']
    y = data['winner']

    # Podział cech na kategoryczne i liczbowe
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numerical_cols = [col for col in X.columns if X[col].dtype != 'object']

    # Kodowanie cech kategorycznych i standaryzacja liczbowych
    if for_nn or for_onehot:
        # One-hot encoding cech kategorycznych
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = ohe.fit_transform(X[categorical_cols])
        # Połączenie cech kategorycznych i liczbowych
        X_num = X[numerical_cols].values
        X = np.hstack([X_cat, X_num])
        # Standaryzacja cech liczbowych
        scaler = StandardScaler()
        X[:, -len(numerical_cols):] = scaler.fit_transform(X[:, -len(numerical_cols):])
    else:
        # Label encoding cech kategorycznych
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        # Standaryzacja cech liczbowych
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Kodowanie etykiet y
    y = LabelEncoder().fit_transform(y)
    
    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=285733, stratify=y)
    return X_train, X_test, y_train, y_test