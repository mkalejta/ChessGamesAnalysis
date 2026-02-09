import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def model_builder(units=128, dropout=0.3, input_dim=100):
    model = Sequential()
    model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.001), input_shape=(input_dim,)))
    model.add(Dropout(dropout))
    model.add(Dense(units//2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network_gridsearch(X_train, X_test, y_train, y_test):
    keras_clf = KerasClassifier(
        model=model_builder,
        epochs=30,
        batch_size=64,
        verbose=0
    )

    param_grid = {
        'model__units': [64, 128, 256],
        'model__dropout': [0.2, 0.3, 0.4],
        'model__input_dim': [X_train.shape[1]],
        'epochs': [30, 50]
    }

    grid = GridSearchCV(
        estimator=keras_clf,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=1
    )

    grid.fit(X_train, y_train)
    os.makedirs('best_params', exist_ok=True)
    with open('best_params/neural_network_gridsearch.txt', 'w') as f:
        f.write('\nBest params: ' + str(grid.best_params_))
        f.write('\nBest score: ' + str(grid.best_score_))
        best_model = grid.best_estimator_
        test_acc = accuracy_score(y_test, np.argmax(best_model.model_.predict(X_test), axis=1))
        f.write('\nTest accuracy: ' + str(test_acc))
    return test_acc