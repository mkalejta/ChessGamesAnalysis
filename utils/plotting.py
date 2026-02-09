import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import learning_curve

def plot_train_test_accuracy(train_acc, test_acc, model_name):
    plt.figure()
    plt.bar(['Train', 'Test'], [train_acc, test_acc], color=['skyblue', 'orange'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy: Train vs Test')
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{model_name.lower().replace(' ', '_')}_train_vs_test.png")
    plt.clf()

def plot_learning_curve(estimator, X, y, model_name, cv=5, scoring='accuracy'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', label="Train")
    plt.plot(train_sizes, test_mean, 'o-', label="Test")
    plt.title(f'Krzywa uczenia: {model_name}')
    plt.xlabel("Liczba próbek treningowych")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.grid()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/learning_curve_{model_name.lower().replace(' ', '_')}.png")
    plt.close()