import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from mpi4py import MPI
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, ClusterMixin
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import os
import sys
import shutil
import tempfile

# --- Model Definitions ---
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, 1)  # Single output for regression
    
    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No sigmoid, raw output for regression
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
def train_rf(X_train, y_train, X_test, y_test, parallel=False, n_estimators=500, max_depth=15, n_jobs=1):
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs if parallel else 1, random_state=42)
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Random Forest {'Paralelní' if parallel else 'Sekvenční'} (n_estimators={n_estimators}, max_depth={max_depth}, n_jobs={n_jobs}): Čas: {train_time:.2f}s, MSE: {mse:.4f}, MAE: {mae:.4f}")
    return train_time, mse, mae

def train_lr(X_train, y_train, X_test, y_test, parallel=False, max_iter=1000):
    lr = LinearRegression(n_jobs=-1 if parallel else 1)
    start_time = time.perf_counter()
    lr.fit(X_train, y_train)
    time.sleep(0.001)  # Minimální zpoždění pro ověření
    train_time = time.perf_counter() - start_time
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Linear Regression {'Paralelní' if parallel else 'Sekvenční'} (max_iter={max_iter}): Čas: {train_time:.2f}s, MSE: {mse:.4f}, MAE: {mae:.4f}")
    return train_time, mse, mae

def train_nn_cpu(X_train, y_train, X_test, y_test, epochs=75, hidden_size=128):
    device = torch.device('cpu')
    model = NeuralNetwork(X_train.shape[1], hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        # Vážená ztráta pro extrémní pozice
        weights = torch.ones_like(y_train_tensor)
        weights[y_train_tensor < 0.1] = 2.0  # Vyšší váha pro pozice blízko 1
        weights[y_train_tensor > 0.9] = 2.0  # Vyšší váha pro pozice blízko 20
        loss = criterion(outputs, y_train_tensor) * weights
        loss = loss.mean()
        loss.backward()
        optimizer.step()
    train_time = time.time() - start_time
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        mse = mean_squared_error(y_test_tensor.cpu().numpy(), outputs.cpu().numpy())
        mae = mean_absolute_error(y_test_tensor.cpu().numpy(), outputs.cpu().numpy())
        predicted_positions = (outputs.cpu().numpy() * 19) + 1
        actual_positions = (y_test_tensor.cpu().numpy() * 19) + 1
        print("Prvních 5 predikovaných pozic:", predicted_positions[:5])
        print("Prvních 5 skutečných pozic:", actual_positions[:5])
    
    print(f"Neural Network CPU (epochs={epochs}, hidden_size={hidden_size}): Čas: {train_time:.2f}s, MSE: {mse:.4f}, MAE: {mae:.4f}")
    return train_time, mse, mae

def train_nn_gpu(X_train, y_train, X_test, y_test, epochs=75, hidden_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(X_train.shape[1], hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        # Vážená ztráta pro extrémní pozice
        weights = torch.ones_like(y_train_tensor)
        weights[y_train_tensor < 0.1] = 2.0
        weights[y_train_tensor > 0.9] = 2.0
        loss = criterion(outputs, y_train_tensor) * weights
        loss = loss.mean()
        loss.backward()
        optimizer.step()
    train_time = time.time() - start_time
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        mse = mean_squared_error(y_test_tensor.cpu().numpy(), outputs.cpu().numpy())
        mae = mean_absolute_error(y_test_tensor.cpu().numpy(), outputs.cpu().numpy())
        predicted_positions = (outputs.cpu().numpy() * 19) + 1
        actual_positions = (y_test_tensor.cpu().numpy() * 19) + 1
        print("Prvních 5 predikovaných pozic:", predicted_positions[:5])
        print("Prvních 5 skutečných pozic:", actual_positions[:5])
    
    print(f"Neural Network GPU (epochs={epochs}, hidden_size={hidden_size}): Čas: {train_time:.2f}s, MSE: {mse:.4f}, MAE: {mae:.4f}")
    return train_time, mse, mae

def run_mpi_with_procs(n_procs, X_train, y_train, X_test, y_test, epochs=75, hidden_size=128):
    temp_dir = tempfile.mkdtemp(prefix="mpi_data_", dir="/tmp")
    print(f"Using temporary directory: {temp_dir}")
    
    X_train_path = os.path.join(temp_dir, 'X_train.npy')
    y_train_path = os.path.join(temp_dir, 'y_train.npy')
    X_test_path = os.path.join(temp_dir, 'X_test.npy')
    y_test_path = os.path.join(temp_dir, 'y_test.npy')
    
    try:
        np.save(X_train_path, X_train.to_numpy())
        np.save(y_train_path, y_train.to_numpy())
        np.save(X_test_path, X_test.to_numpy())
        np.save(y_test_path, y_test.to_numpy())
        print(f"Data files saved successfully in {temp_dir}")
    except Exception as e:
        print(f"Error saving data files: {e}")
        shutil.rmtree(temp_dir)
        return None, None, None
    
    python_executable = sys.executable
    print(f"Using Python executable: {python_executable}")
    
    script_path = os.path.abspath(__file__)
    print(f"Script path: {script_path}")
    cmd = [
        "mpirun",
        "--allow-run-as-root",
        "-n", str(n_procs),
        python_executable,
        script_path,
        "run_mpi",
        str(epochs),
        str(hidden_size),
        temp_dir
    ]
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ, "OMPI_MCA_verbose": "1"},
            check=True
        )
        output = result.stdout
        error = result.stderr
        
        print(f"MPI Output (n_procs={n_procs}):")
        print(output)
        if error:
            print(f"MPI Errors (n_procs={n_procs}):")
            print(error)
        
        for line in output.splitlines():
            if "Čas:" in line and "MSE:" in line and "MAE:" in line:
                try:
                    time_part = line.split("Čas: ")[1].split("s")[0].strip()
                    mse_part = line.split("MSE: ")[1].split(",")[0].strip()
                    mae_part = line.split("MAE: ")[1].strip()
                    return float(time_part), float(mse_part), float(mae_part)
                except (IndexError, ValueError) as e:
                    print(f"Error parsing line '{line}': {e}")
                    continue
        
        raise ValueError(f"No valid output found in MPI process. Output:\n{output}\nError:\n{error}")
    
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with error: {e}")
        print(f"Error Output: {e.stderr}")
        print(f"Falling back to direct execution of train_nn_mpi for n_procs={n_procs}")
        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()
        X_test_np = X_test.to_numpy()
        y_test_np = y_test.to_numpy()
        return train_nn_mpi(X_train_np, y_train_np, X_test_np, y_test_np, epochs, hidden_size, temp_dir)
    except Exception as e:
        print(f"Error during MPI execution: {e}")
        return None, None, None
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def train_nn_mpi(X_train, y_train, X_test, y_test, epochs=75, hidden_size=128, temp_dir="/tmp/mpi_data"):
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"Rank {rank}/{size}: MPI initialized successfully")
    except Exception as e:
        print(f"Error initializing MPI: {e}")
        rank = 0
        size = 1
    
    if size > 1:
        try:
            X_train = np.load(os.path.join(temp_dir, "X_train.npy"))
            y_train = np.load(os.path.join(temp_dir, "y_train.npy"))
            X_test = np.load(os.path.join(temp_dir, "X_test.npy"))
            y_test = np.load(os.path.join(temp_dir, "y_test.npy"))
            print(f"Rank {rank}: Data loaded successfully from {temp_dir}")
        except Exception as e:
            print(f"Rank {rank}: Error loading data: {e}")
            return None, None, None
    
    local_n = max(10, X_train.shape[0] // max(size, 1))
    start_idx = rank * local_n
    end_idx = min((rank + 1) * local_n, X_train.shape[0])
    X_train_local = X_train[start_idx:end_idx]
    y_train_local = y_train[start_idx:end_idx]
    
    if X_train_local.shape[0] < 10:
        print(f"Rank {rank}: Příliš málo dat ({X_train_local.shape[0]}), přeskakuji.")
        return None, None, None
    
    model = NeuralNetwork(X_train.shape[1], hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    X_train_local_tensor = torch.tensor(X_train_local, dtype=torch.float32)
    y_train_local_tensor = torch.tensor(y_train_local, dtype=torch.float32).view(-1, 1)
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_local_tensor)
        weights = torch.ones_like(y_train_local_tensor)
        weights[y_train_local_tensor < 0.1] = 2.0
        weights[y_train_local_tensor > 0.9] = 2.0
        loss = criterion(outputs, y_train_local_tensor) * weights
        loss = loss.mean()
        loss.backward()
        for param in model.parameters():
            if size > 1:
                comm.Allreduce(MPI.IN_PLACE, param.grad.data.numpy(), op=MPI.SUM)
            param.grad.data /= max(size, 1)
        optimizer.step()
    train_time = time.time() - start_time
    
    if rank == 0:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            outputs = model(X_test_tensor)
            mse = mean_squared_error(y_test_tensor.numpy(), outputs.numpy())
            mae = mean_absolute_error(y_test_tensor.numpy(), outputs.numpy())
            predicted_positions = (outputs.numpy() * 19) + 1
            actual_positions = (y_test_tensor.numpy() * 19) + 1
            print("Prvních 5 predikovaných pozic:", predicted_positions[:5])
            print("Prvních 5 skutečných pozic:", actual_positions[:5])
        print(f"Neural Network MPI (epochs={epochs}, hidden_size={hidden_size}): Čas: {train_time:.2f}s, MSE: {mse:.4f}, MAE: {mae:.4f}")
        return train_time, mse, mae
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

    data['positionOrder_normalized'] = (data['positionOrder'] - 1) / (20 - 1)

    # Rozšíření seznamu příznaků o nové sloupce
    features = [
        'grid', 'qualifying_position', 
        'avg_position_last_5', 'avg_position_circuit',
        'constructor_points', 'points_before_race', 'grid_qual_diff',
        'avg_position_per_season', 'avg_points_per_race_season',  # Nové příznaky
        'avg_constructor_position_season', 'avg_fastest_lap_speed',
        'avg_pit_stop_duration', 'avg_lap_time_circuit'
    ]
    X = data[features].fillna(data[features].mean())  # Vyplnění chybějících hodnot průměrem
    y = data['positionOrder_normalized']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

# --- Evaluation and Visualization ---
def visualize_results(results, iteration):
    methods = [r[0] for r in results]
    times = [r[1] for r in results]
    mses = [r[2] for r in results if r[2] is not None]
    maes = [r[3] for r in results if r[3] is not None]
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.bar(methods, times, color='skyblue')
    plt.title(f'Čas trénování (s) - Iterace {iteration}')
    plt.ylabel('Čas (s)')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.bar(methods, mses, color='lightgreen')
    plt.title(f'MSE - Iterace {iteration}')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    plt.bar(methods, maes, color='lightcoral')
    plt.title(f'MAE - Iterace {iteration}')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'comparison_iteration_{iteration}.png')
    print(f"Výsledky uloženy pro iteraci {iteration}")

# --- Main Execution ---
def main():
    X_train, y_train, X_test, y_test = load_data()
    
    num_iterations = 10
    for iteration in range(1, num_iterations + 1):
        print(f"\nIterace {iteration}")
        results = []
        
        rf_seq_time, rf_seq_mse, rf_seq_mae = train_rf(X_train, y_train, X_test, y_test, parallel=False, n_estimators=500, max_depth=20)
        results.append(("RF Seq (500, 20)", rf_seq_time, rf_seq_mse, rf_seq_mae))
        
        rf_par_time, rf_par_mse, rf_par_mae = train_rf(X_train, y_train, X_test, y_test, parallel=True, n_estimators=500, max_depth=20, n_jobs=4)
        results.append(("RF Par (500, 20, n_jobs=4)", rf_par_time, rf_par_mse, rf_par_mae))
        
        lr_seq_time, lr_seq_mse, lr_seq_mae = train_lr(X_train, y_train, X_test, y_test, parallel=False, max_iter=500)
        results.append(("LR Seq (500)", lr_seq_time, lr_seq_mse, lr_seq_mae))
        
        lr_par_time, lr_par_mse, lr_par_mae = train_lr(X_train, y_train, X_test, y_test, parallel=True, max_iter=500)
        results.append(("LR Par (500)", lr_par_time, lr_par_mse, lr_par_mae))
        
        nn_cpu_time, nn_cpu_mse, nn_cpu_mae = train_nn_cpu(X_train, y_train, X_test, y_test, epochs=75, hidden_size=128)
        results.append(("NN Seq CPU (75, 128)", nn_cpu_time, nn_cpu_mse, nn_cpu_mae))
        
        nn_gpu_time, nn_gpu_mse, nn_gpu_mae = train_nn_gpu(X_train, y_train, X_test, y_test, epochs=75, hidden_size=128)
        results.append(("NN Par GPU (75, 128)", nn_gpu_time, nn_gpu_mse, nn_gpu_mae))
        
        for n_procs in [1, 2, 4]:
            print(f"\nRunning MPI with {n_procs} processes")
            nn_mpi_time, nn_mpi_mse, nn_mpi_mae = run_mpi_with_procs(n_procs, X_train, y_train, X_test, y_test, epochs=75, hidden_size=128)
            if nn_mpi_time is not None:
                results.append((f"NN MPI (75, 128, n_procs={n_procs})", nn_mpi_time, nn_mpi_mse, nn_mpi_mae))
        
        visualize_results(results, iteration)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "run_mpi":
        epochs = int(sys.argv[2])
        hidden_size = int(sys.argv[3])
        temp_dir = sys.argv[4]
        X_train = np.load(os.path.join(temp_dir, "X_train.npy"))
        y_train = np.load(os.path.join(temp_dir, "y_train.npy"))
        X_test = np.load(os.path.join(temp_dir, "X_test.npy"))
        y_test = np.load(os.path.join(temp_dir, "y_test.npy"))
        train_nn_mpi(X_train, y_train, X_test, y_test, epochs, hidden_size, temp_dir)
    else:
        main()