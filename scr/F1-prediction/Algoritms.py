import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from dask_ml.model_selection import GridSearchCV
import dask.dataframe as dd
import time
import matplotlib.pyplot as plt

data = pd.read_csv('../Dataset/output/2024.csv')
# Vytvoření cílové proměnné
data['winner'] = (data['positionOrder'] == 1).astype(int)

# Výběr příznaků
features = ['grid', 'points', 'qualifying_position', 
            'avg_position_last_5', 'avg_position_circuit',
            'constructor_points','grid_qual_diff','position']
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
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if not parallel:
        model.fit(X_train, y_train,epochs=100, batch_size=32, verbose=0)
    else:
        best_acc = 0
        best_model = None
        for units in [32, 64, 128]:
            temp_model = Sequential([
                Dense(units, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(units//2, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            temp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            temp_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
            loss, acc = temp_model.evaluate(X_test, y_test, verbose=0)
            if acc > best_acc:
                best_acc = acc
                best_model = temp_model
        model = best_model
    train_time = time.time() - start_time
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} {'Paralelní' if parallel else 'Sekvenční'}: Čas: {train_time:.2f}s, Přesnost: {accuracy:.4f}, F1-skóre: {f1:.4f}")
    return train_time, accuracy, f1

# Sekvenční trénování
rf_seq = RandomForestClassifier(n_estimators=1000,n_jobs=1, random_state=42)
rf_seq_time, rf_seq_acc, rf_seq_f1 = train_and_evaluate(rf_seq, X_train, X_test, y_train, y_test, "Random Forest")

lr_seq = LogisticRegression(max_iter=1000, random_state=42)
lr_seq_time, lr_seq_acc, lr_seq_f1 = train_and_evaluate(lr_seq, X_train, X_test, y_train, y_test, "Logistic Regression")

nn_seq_time, nn_seq_acc, nn_seq_f1 = train_nn(X_train, X_test, y_train, y_test, "Neural Network")

# Paralelní trénování
rf_par = RandomForestClassifier(n_estimators=1000,n_jobs=-1, random_state=42)
rf_par_time, rf_par_acc, rf_par_f1 = train_and_evaluate(rf_par, X_train, X_test, y_train, y_test, "Random Forest", parallel=True)

param_grid = {'C': [0.1, 1, 10]}
lr_dask = LogisticRegression(max_iter=1000, random_state=42)
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

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(models, times, color='skyblue')
plt.title('Čas trénování (s)')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
plt.bar(models, accuracies, color='lightgreen')
plt.title('Přesnost')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('training_comparison02.png')