import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
import keras
from keras import layers

from dask_ml.model_selection import GridSearchCV
import dask.dataframe as dd
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

print("TensorFlow version:", tf.config.list_physical_devices('GPU'))
# Načtení datasetu a simulace většího datasetu opakováním 10x
data = pd.read_csv('../Dataset/output/2024.csv')
data = pd.concat([data] * 10, ignore_index=True)

# Vytvoření cílové proměnné
data['winner'] = (data['positionOrder'] == 1).astype(int)

# Výběr příznaků
features = ['grid', 'points', 'qualifying_position', 
            'avg_position_last_5', 'avg_position_circuit',
            'constructor_points', 'grid_qual_diff', 'position']
X = data[features].fillna(data[features].mean())
y = data['winner']

# Škálování příznaků
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Funkce pro trénování scikit-learn modelů
def train_and_evaluate(model, X_train, X_test, y_train, y_test, name, parallel=False):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} {'Paralelní' if parallel else 'Sekvenční'}: Čas: {train_time:.2f}s, Přesnost: {accuracy:.4f}, F1-skóre: {f1:.4f}")
    return train_time, accuracy, f1

# Funkce pro trénování neuronové sítě
def train_nn(X_train, X_test, y_train, y_test, name, parallel=False):
    start_time = time.time()
    if not parallel:
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    else:
        def train_model(units):
            model = keras.Sequential([
                keras.D
                keras.Dense(units//2, activation='relu'),
                keras.Dense(units//4, activation='relu'),
                keras.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            return model
        models = Parallel(n_jobs=-1)(delayed(train_model)(units) for units in [64, 128, 256])
        best_model = max(models, key=lambda m: m.evaluate(X_test, y_test, verbose=0)[1])
        model = best_model
    train_time = time.time() - start_time
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} {'Paralelní' if parallel else 'Sekvenční'}: Čas: {train_time:.2f}s, Přesnost: {accuracy:.4f}, F1-skóre: {f1:.4f}")
    return train_time, accuracy, f1

# Sekvenční trénování
rf_seq = RandomForestClassifier(n_estimators=1000, max_depth=20, n_jobs=1, random_state=42)
rf_seq_time, rf_seq_acc, rf_seq_f1 = train_and_evaluate(rf_seq, X_train, X_test, y_train, y_test, "Random Forest")

lr_seq = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
lr_seq_time, lr_seq_acc, lr_seq_f1 = train_and_evaluate(lr_seq, X_train, X_test, y_train, y_test, "Logistic Regression")

nn_seq_time, nn_seq_acc, nn_seq_f1 = train_nn(X_train, X_test, y_train, y_test, "Neural Network")

# Paralelní trénování
rf_par = RandomForestClassifier(n_estimators=1000, max_depth=20, n_jobs=-1, random_state=42)
rf_par_time, rf_par_acc, rf_par_f1 = train_and_evaluate(rf_par, X_train, X_test, y_train, y_test, "Random Forest", parallel=True)

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
lr_dask = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
grid_search = GridSearchCV(lr_dask, param_grid, cv=5, n_jobs=-1)
ddf = dd.from_pandas(pd.concat([X_train, y_train], axis=1), npartitions=4)
start_time = time.time()
grid_search.fit(ddf[features], ddf['winner'])
dask_lr_time = time.time() - start_time
y_pred = grid_search.predict(X_test)
lr_dask_acc = accuracy_score(y_test, y_pred)
lr_dask_f1 = f1_score(y_test, y_pred)
print(f"Logistic Regression Dask: Čas: {dask_lr_time:.2f}s, Přesnost: {lr_dask_acc:.4f}, F1-skóre: {lr_dask_f1:.4f}")

nn_par_time, nn_par_acc, nn_par_f1 = train_nn(X_train, X_test, y_train, y_test, "Neural Network", parallel=True)

# Vizualizace
models = ['RF Seq', 'RF Par', 'LR Seq', 'LR Dask', 'NN Seq', 'NN Par']
times = [rf_seq_time, rf_par_time, lr_seq_time, dask_lr_time, nn_seq_time, nn_par_time]
accuracies = [rf_seq_acc, rf_par_acc, lr_seq_acc, lr_dask_acc, nn_seq_acc, nn_par_acc]
f1_scores = [rf_seq_f1, rf_par_f1, lr_seq_f1, lr_dask_f1, nn_seq_f1, nn_par_f1]

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.bar(models, times, color='skyblue')
plt.title('Čas trénování (s)')
plt.xticks(rotation=45)
plt.subplot(1, 3, 2)
plt.bar(models, accuracies, color='lightgreen')
plt.title('Přesnost')
plt.xticks(rotation=45)
plt.subplot(1, 3, 3)
plt.bar(models, f1_scores, color='lightcoral')
plt.title('F1-skóre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('training_comparison03.png')