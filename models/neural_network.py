from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from utils.plotting import plot_train_test_accuracy
import matplotlib.pyplot as plt
import os

def train_neural_network(X_train, X_test, y_train, y_test):
    model_path = "neural_network_model.h5"

    if os.path.exists(model_path):
        # Załadowanie istniejącego modelu
        model = load_model(model_path)
        print("Załadowano istniejący model. Dotrenowywanie...")
    else:
        # Utworzenie nowego modelu
        model = Sequential([
            Dense(64, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(3, activation='softmax')
        ])
    
    # Kompilacja modelu
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    # Zapisanie modelu do pliku
    model.save(model_path)

    # Obliczenie accuracy na train i test
    train_acc = accuracy_score(y_train, model.predict(X_train).argmax(axis=1))
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Zapisanie wykresu
    plot_train_test_accuracy(train_acc, test_acc, "Neural Network")
    plot_nn_learning_curve(history, "Neural Network")

    return test_acc

def plot_nn_learning_curve(history, model_name):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'Krzywa uczenia: {model_name}')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()
    plt.grid()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/learning_curve_{model_name.lower().replace(' ', '_')}.png")
    plt.close()