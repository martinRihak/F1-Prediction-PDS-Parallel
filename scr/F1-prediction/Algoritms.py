import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from dask_ml.model_selection import GridSearchCV
import dask.dataframe as dd
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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

# Funkce pro trénování scikit-learn modelů (beze změny)
def train_and_evaluate(model, X_train, X_test, y_train, y_test, name, parallel=False):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} {'Paralelní' if parallel else 'Sekvenční'}: Čas: {train_time:.2f}s, Přesnost: {accuracy:.4f}, F1-skóre: {f1:.4f}")
    return train_time, accuracy, f1

# Definice neuronové sítě v PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Funkce pro trénování neuronové sítě v PyTorch
def train_nn(X_train, X_test, y_train, y_test, name, parallel=False):
    start_time = time.time()
    if not parallel:
        # Sekvenční trénování
        model = NeuralNetwork(X_train.shape[1])
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = torch.optim.Adam(model.parameters())
        
        # Převod dat na PyTorch tensory
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        
        # Trénovací smyčka
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
    else:
        # Paralelní trénování s různými velikostmi vrstev
        def train_model(units):
            model = NeuralNetwork(X_train.shape[1])
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters())
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            return model
        
        models = Parallel(n_jobs=-1)(delayed(train_model)(units) for units in [128, 256, 512])
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        best_model = max(models, key=lambda m: (m(X_test_tensor) > 0.5).float().eq(y_test_tensor).float().mean().item())
        model = best_model
    
    # Výpočet času a evaluace
    train_time = time.time() - start_time
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        outputs = model(X_test_tensor)
        y_pred = (outputs > 0.5).float()
        accuracy = (y_pred == y_test_tensor).float().mean().item()
        f1 = f1_score(y_test_tensor.numpy(), y_pred.numpy())
    print(f"{name} {'Paralelní' if parallel else 'Sekvenční'}: Čas: {train_time:.2f}s, Přesnost: {accuracy:.4f}, F1-skóre: {f1:.4f}")
    return train_time, accuracy, f1

# Sekvenční trénování
rf_seq = RandomForestClassifier(n_estimators=2000, max_depth=30, n_jobs=1, random_state=42)
rf_seq_time, rf_seq_acc, rf_seq_f1 = train_and_evaluate(rf_seq, X_train, X_test, y_train, y_test, "Random Forest")

lr_seq = LogisticRegression(solver='saga', max_iter=2000, random_state=42)
lr_seq_time, lr_seq_acc, lr_seq_f1 = train_and_evaluate(lr_seq, X_train, X_test, y_train, y_test, "Logistic Regression")

nn_seq_time, nn_seq_acc, nn_seq_f1 = train_nn(X_train, X_test, y_train, y_test, "Neural Network")

# Paralelní trénování
rf_par = RandomForestClassifier(n_estimators=2000, max_depth=30, n_jobs=-1, random_state=42)
rf_par_time, rf_par_acc, rf_par_f1 = train_and_evaluate(rf_par, X_train, X_test, y_train, y_test, "Random Forest", parallel=True)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2', 'elasticnet'], 'l1_ratio': [0.0, 0.5, 1.0]}
lr_dask = LogisticRegression(solver='saga', max_iter=2000, random_state=42)
grid_search = GridSearchCV(lr_dask, param_grid, cv=5, n_jobs=-1)
ddf = dd.from_pandas(pd.concat([X_train, y_train], axis=1), npartitions=8)
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
plt.title('Čas trénování (s)', fontsize=12)
plt.ylabel('Čas (s)', fontsize=10)
plt.xticks(rotation=45)
plt.subplot(1, 3, 2)
plt.bar(models, accuracies, color='lightgreen')
plt.title('Přesnost', fontsize=12)
plt.ylabel('Přesnost', fontsize=10)
plt.xticks(rotation=45)
plt.subplot(1, 3, 3)
plt.bar(models, f1_scores, color='lightcoral')
plt.title('F1-skóre', fontsize=12)
plt.ylabel('F1-skóre', fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('training_comparison04.png')