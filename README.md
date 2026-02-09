<h1 align="center">Chess Games Analysis</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=flat-square&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
</p>

<p align="center">
  Project developed for the <b>Computational Intelligence</b> university course.<br/>
  Predicting the outcome of chess games (white wins, black wins, or draw) using various machine learning models with different architectures, activation functions, and hyperparameters.
</p>

---

## About

The dataset (`games.csv`) contains chess games played on Lichess. Each game record includes player ratings, opening classification (ECO codes), move sequences, time controls, and the final result.

The project tackles a **multi-class classification** problem — predicting the winner of a chess game based on:

- The first 30 half-moves (15 moves per side)
- Opening ECO code and name
- Number of turns, time control, game duration
- Whether the game was rated
- Victory status (checkmate, resignation, timeout, draw)
- Temporal features (hour and weekday of game start)

---

## Models Implemented

| Model | Description | Key Hyperparameters |
|---|---|---|
| **Neural Network** | Keras Sequential model with Dense layers | ReLU + Softmax activations, L2 regularization, Dropout (0.3), Adam optimizer, EarlyStopping |
| **Random Forest** | Ensemble of 500 decision trees | max_depth=20, balanced class weights |
| **XGBoost** | Gradient boosted trees | 500 estimators, max_depth=8, learning_rate=0.1, colsample_bytree=0.7 |
| **KNN** | K-Nearest Neighbors | k=7, distance-weighted voting |

### Neural Network Architecture

```
Input Layer  →  Dense(64, activation='relu', L2=0.001)  →  Dropout(0.3)
             →  Dense(32, activation='relu', L2=0.001)  →  Dropout(0.3)
             →  Dense(3, activation='softmax')
```

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Callbacks:** EarlyStopping (patience=10, restore best weights)
- **Training:** 30 epochs, batch size 64, 20% validation split

---

## Hyperparameter Tuning (GridSearchCV)

Each model has a dedicated GridSearchCV module that systematically searches over a parameter grid using 3-fold cross-validation:

| Model | Tuned Parameters | Best Test Accuracy |
|---|---|---|
| Neural Network | units (64/128/256), dropout (0.2/0.3/0.4), epochs (30/50) | 58.5% |
| Random Forest | n_estimators (100/300/500), max_depth (10/20/30), min_samples_split/leaf | 57.6% |
| **XGBoost** | n_estimators, max_depth, learning_rate, subsample, colsample_bytree | **86.3%** |
| KNN | n_neighbors (3/5/7/9), weights (uniform/distance) | 51.5% |

---

## Association Rules Mining

The project uses the **Apriori algorithm** (via mlxtend) to discover association rules between chess openings (ECO codes) and game outcomes, ranked by lift metric.

---

## Project Structure

```
chess-games-analysis/
├── chess.py                        # Main entry point — trains all models
├── association_rules.py            # Apriori-based opening → winner rules
├── games.csv                       # Lichess games dataset
├── requirements.txt                # Python dependencies
├── neural_network_model.h5         # Saved Keras model weights
├── models/
│   ├── neural_network.py           # Keras Sequential neural network
│   ├── random_forest.py            # scikit-learn Random Forest
│   ├── xgboost_model.py            # XGBoost classifier
│   └── knn_model.py                # K-Nearest Neighbors classifier
├── grid_search/
│   ├── neural_network_gridsearch.py
│   ├── random_forest_gridsearch.py
│   ├── xgboost_gridsearch.py
│   └── knn_gridsearch.py
├── utils/
│   ├── preprocessing.py            # Data loading, feature engineering, encoding
│   └── plotting.py                 # Learning curves & accuracy bar charts
├── best_params/                    # GridSearchCV results (best params + scores)
├── plots/                          # Generated visualizations
│   ├── learning_curve_*.png
│   ├── *_train_vs_test.png
│   ├── rozklad_liczby_ruchow.png
│   └── rozklad_ocen_graczy.png
└── Chess_Games.pdf                 # Project report
```

---

## Technologies

| Technology | Purpose |
|---|---|
| **Python 3.12** | Programming language |
| **TensorFlow / Keras** | Neural network construction, training, and evaluation |
| **scikit-learn** | Random Forest, KNN, GridSearchCV, preprocessing (OneHotEncoder, LabelEncoder, StandardScaler), learning curves |
| **SciKeras** | Keras wrapper for scikit-learn compatibility (used in GridSearchCV for neural networks) |
| **XGBoost** | Gradient boosted tree classifier |
| **mlxtend** | Apriori algorithm and association rules mining |
| **pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib** | Plotting learning curves and accuracy comparisons |
| **seaborn** | Statistical data visualization |

---

## Quickstart

### 1. Clone the repository

```bash
git clone <repository-url>
cd chess-games-analysis
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> TensorFlow requires Python 3.9–3.12. Make sure your Python version is compatible.

### 4. Run model training

```bash
python chess.py
```

This will:
1. Load and preprocess the chess games dataset (`games.csv`)
2. Engineer features from the first 15 moves per side, openings, time controls, and temporal data
3. Apply One-Hot Encoding for categorical features and StandardScaler for numerical ones
4. Train all four models (Neural Network, Random Forest, XGBoost, KNN)
5. Generate learning curve plots and train-vs-test accuracy charts in the `plots/` directory
6. Print a comparison of test accuracies for all models

### 5. Run hyperparameter tuning (optional)

To run GridSearchCV for all models, uncomment the grid search lines in `chess.py` (lines 28–35) and re-run:

```bash
python chess.py
```

Results (best parameters and scores) will be saved to the `best_params/` directory.

### 6. Run association rules mining (optional)

```bash
python association_rules.py
```

This discovers and prints the top 20 association rules between chess openings and game outcomes, saving results to `association_rules_results.csv`.

---

## Results

The best-performing model was **XGBoost** achieving **86.3% test accuracy** after hyperparameter tuning via GridSearchCV. Generated plots (learning curves, accuracy comparisons) can be found in the `plots/` directory.
