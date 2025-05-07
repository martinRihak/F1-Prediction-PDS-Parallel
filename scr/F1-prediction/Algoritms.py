import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClusterMixin
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd

# --- Model Definitions ---
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

class ParallelKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=300, n_jobs=-1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_jobs = n_jobs
    
    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = Parallel(n_jobs=self.n_jobs)(delayed(np.mean)(X[labels == k], axis=0) 
                                                        for k in range(self.n_clusters))
            self.centroids = np.array(new_centroids)
        return self
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# --- Training Functions ---
def train_rf(X_train, y_train, X_test, y_test, parallel=False, n_estimators=1000, max_depth=20):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1 if parallel else 1, random_state=42)
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Random Forest {'Paralelní' if parallel else 'Sekvenční'} (n_estimators={n_estimators}, max_depth={max_depth}): Čas: {train_time:.2f}s, Přesnost: {accuracy:.4f}, F1-skóre: {f1:.4f}")
    return train_time, accuracy, f1

def train_lr(X_train, y_train, X_test, y_test, parallel=False, max_iter=1000):
    lr = LogisticRegression(solver='saga', max_iter=max_iter, n_jobs=-1 if parallel else 1, random_state=42)
    start_time = time.time()
    lr.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Logistic Regression {'Paralelní' if parallel else 'Sekvenční'} (max_iter={max_iter}): Čas: {train_time:.2f}s, Přesnost: {accuracy:.4f}, F1-skóre: {f1:.4f}")
    return train_time, accuracy, f1

def train_nn_cpu(X_train, y_train, X_test, y_test, epochs=50, hidden_size=256):
    device = torch.device('cpu')
    model = NeuralNetwork(X_train.shape[1], hidden_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start_time
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_pred = (outputs > 0.5).float()
        accuracy = (y_pred == y_test_tensor).float().mean().item()
        f1 = f1_score(y_test_tensor.cpu().numpy(), y_pred.cpu().numpy())
    
    print(f"Neural Network CPU (epochs={epochs}, hidden_size={hidden_size}): Čas: {train_time:.2f}s, Přesnost: {accuracy:.4f}, F1-skóre: {f1:.4f}")
    return train_time, accuracy, f1

def train_nn_gpu(X_train, y_train, X_test, y_test, epochs=50, hidden_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(X_train.shape[1], hidden_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start_time
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_pred = (outputs > 0.5).float()
        accuracy = (y_pred == y_test_tensor).float().mean().item()
        f1 = f1_score(y_test_tensor.cpu().numpy(), y_pred.cpu().numpy())
    
    print(f"Neural Network GPU (epochs={epochs}, hidden_size={hidden_size}): Čas: {train_time:.2f}s, Přesnost: {accuracy:.4f}, F1-skóre: {f1:.4f}")
    return train_time, accuracy, f1

def train_nn_mpi(X_train, y_train, X_test, y_test, epochs=50, hidden_size=256):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_n = X_train.shape[0] // size
    X_train_local = X_train[rank*local_n:(rank+1)*local_n]
    y_train_local = y_train[rank*local_n:(rank+1)*local_n]
    
    model = NeuralNetwork(X_train.shape[1], hidden_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    X_train_local_tensor = torch.tensor(X_train_local.values, dtype=torch.float32)
    y_train_local_tensor = torch.tensor(y_train_local.values, dtype=torch.float32).view(-1, 1)
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_local_tensor)
        loss = criterion(outputs, y_train_local_tensor)
        loss.backward()
        for param in model.parameters():
            comm.Allreduce(MPI.IN_PLACE, param.grad.data.numpy(), op=MPI.SUM)
            param.grad.data /= size
        optimizer.step()
    train_time = time.time() - start_time
    
    if rank == 0:
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            outputs = model(X_test_tensor)
            y_pred = (outputs > 0.5).float()
            accuracy = (y_pred == y_test_tensor).float().mean().item()
            f1 = f1_score(y_test_tensor.numpy(), y_pred.numpy())
        print(f"Neural Network MPI (epochs={epochs}, hidden_size={hidden_size}): Čas: {train_time:.2f}s, Přesnost: {accuracy:.4f}, F1-skóre: {f1:.4f}")
        return train_time, accuracy, f1
    return None, None, None

def train_parallel_kmeans(X, n_clusters=3):
    start_time = time.time()
    kmeans = ParallelKMeans(n_clusters=n_clusters, max_iter=300, n_jobs=-1)
    kmeans.fit(X)
    train_time = time.time() - start_time
    print(f"Parallel K-means (n_clusters={n_clusters}): Čas: {train_time:.2f}s")
    return train_time

# --- Data Loading and Preprocessing ---
def load_data():
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
    return X_train, y_train, X_test, y_test

# --- Evaluation and Visualization ---
def visualize_results(results):
    methods = [r[0] for r in results]
    times = [r[1] for r in results]
    accuracies = [r[2] for r in results if r[2] is not None]
    f1_scores = [r[3] for r in results if r[3] is not None]
    acc_methods = [m for m, a in zip(methods, [r[2] for r in results]) if a is not None]
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.bar(methods, times, color='skyblue')
    plt.title('Čas trénování (s)')
    plt.ylabel('Čas (s)')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.bar(acc_methods, accuracies, color='lightgreen')
    plt.title('Přesnost')
    plt.ylabel('Přesnost')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    plt.bar(acc_methods, f1_scores, color='lightcoral')
    plt.title('F1-skóre')
    plt.ylabel('F1-skóre')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('comparison.png')
    print("Výsledky uloženy jako 'comparison.png'")

# --- Main Execution ---
def main():
    X_train, y_train, X_test, y_test = load_data()
    
    results = []
    
    # Random Forest Sequential (different settings)
    rf_seq_time, rf_seq_acc, rf_seq_f1 = train_rf(X_train, y_train, X_test, y_test, parallel=False, n_estimators=500, max_depth=10)
    results.append(("RF Seq (500, 10)", rf_seq_time, rf_seq_acc, rf_seq_f1))
    
    # Random Forest Parallel (different settings)
    rf_par_time, rf_par_acc, rf_par_f1 = train_rf(X_train, y_train, X_test, y_test, parallel=True, n_estimators=2000, max_depth=30)
    results.append(("RF Par (2000, 30)", rf_par_time, rf_par_acc, rf_par_f1))
    
    # Logistic Regression Sequential (different settings)
    lr_seq_time, lr_seq_acc, lr_seq_f1 = train_lr(X_train, y_train, X_test, y_test, parallel=False, max_iter=500)
    results.append(("LR Seq (500)", lr_seq_time, lr_seq_acc, lr_seq_f1))
    
    # Logistic Regression Parallel (different settings)
    lr_par_time, lr_par_acc, lr_par_f1 = train_lr(X_train, y_train, X_test, y_test, parallel=True, max_iter=2000)
    results.append(("LR Par (2000)", lr_par_time, lr_par_acc, lr_par_f1))
    
    # Neural Network Sequential (CPU) (different settings)
    nn_cpu_time, nn_cpu_acc, nn_cpu_f1 = train_nn_cpu(X_train, y_train, X_test, y_test, epochs=100, hidden_size=256)
    results.append(("NN Seq CPU (100, 256)", nn_cpu_time, nn_cpu_acc, nn_cpu_f1))
    
    # Neural Network Parallel (GPU) (different settings)
    nn_gpu_time, nn_gpu_acc, nn_gpu_f1 = train_nn_gpu(X_train, y_train, X_test, y_test, epochs=100, hidden_size=256)
    results.append(("NN Par GPU (100, 256)", nn_gpu_time, nn_gpu_acc, nn_gpu_f1))
    
    # Neural Network MPI (different settings)
    nn_mpi_time, nn_mpi_acc, nn_mpi_f1 = train_nn_mpi(X_train, y_train, X_test, y_test, epochs=75, hidden_size=192)
    if nn_mpi_time is not None:
        results.append(("NN MPI (75, 192)", nn_mpi_time, nn_mpi_acc, nn_mpi_f1))
    
    # Parallel K-means (different settings)
    kmeans_time = train_parallel_kmeans(X_train.values, n_clusters=5)
    results.append(("Parallel K-means (5)", kmeans_time, None, None))
    
    visualize_results(results)

if __name__ == "__main__":
    main()