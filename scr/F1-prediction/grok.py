import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import subprocess

# Simulované funkce pro načtení dat a trénování (nahraďte vlastní implementací)
def load_data():
    # Simulace načtení dat
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)
    return X_train, y_train, X_test, y_test

def train_rf(X_train, y_train, X_test, y_test, parallel=False, n_jobs=1):
    start_time = time.time()
    clf = RandomForestClassifier(n_jobs=n_jobs if parallel else 1)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    f1 = 0.85  # Simulované F1 skóre
    return time.time() - start_time, acc, f1

def train_lr(X_train, y_train, X_test, y_test):
    start_time = time.time()
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    f1 = 0.82  # Simulované F1 skóre
    return time.time() - start_time, acc, f1

def train_nn_mpi(X_train, y_train, X_test, y_test, n_procs):
    # Simulace MPI trénování (nahraďte skutečnou MPI implementací)
    start_time = time.time()
    # Předpokládáme, že skript Algorithms.py obsahuje MPI implementaci
    result = subprocess.run(['mpirun', '-n', str(n_procs), 'python', 'Algorithms.py'], capture_output=True, text=True)
    acc = 0.90  # Simulovaná přesnost
    f1 = 0.88  # Simulované F1 skóre
    return time.time() - start_time, acc, f1

def visualize_results(results, suffix):
    methods, times, accuracies, f1_scores = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.bar(methods, times, color='skyblue')
    plt.title(f'Čas trénování – {suffix}')
    plt.ylabel('Čas (s)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'output/comparison_{suffix}.png')
    plt.close()

def main():
    # Vytvoření složky output
    if not os.path.exists('output'):
        os.makedirs('output')

    # 1. Opakování měření (10krát)
    for run in range(1, 11):
        print(f"Spouštím opakování {run}")
        X_train, y_train, X_test, y_test = load_data()
        results = []

        # Základní trénování
        rf_time, rf_acc, rf_f1 = train_rf(X_train, y_train, X_test, y_test, parallel=False)
        results.append(("RF (serial)", rf_time, rf_acc, rf_f1))

        lr_time, lr_acc, lr_f1 = train_lr(X_train, y_train, X_test, y_test)
        results.append(("LR", lr_time, lr_acc, lr_f1))

        # Uložení výsledků
        visualize_results(results, f"run{run}")
        with open(f'output/results_run{run}.txt', 'w') as f:
            for method, time_val, acc, f1 in results:
                f.write(f"{method}: Čas: {time_val:.2f}s, Přesnost: {acc:.4f}, F1-skóre: {f1:.4f}\n")

    # 2. Testování škálovatelnosti – různý počet vláken
    print("Testuji škálovatelnost s různým počtem vláken")
    scalability_results = []
    for n_jobs in [1, 2, 4, 8]:
        X_train, y_train, X_test, y_test = load_data()
        rf_time, rf_acc, rf_f1 = train_rf(X_train, y_train, X_test, y_test, parallel=True, n_jobs=n_jobs)
        scalability_results.append((f"RF n_jobs={n_jobs}", rf_time, rf_acc, rf_f1))
    visualize_results(scalability_results, "scalability_threads")

    # 3. Komplexní analýza MPI – různý počet procesů
    print("Testuji MPI s různým počtem procesů")
    mpi_results = []
    for n_procs in [1, 2, 4, 8]:
        X_train, y_train, X_test, y_test = load_data()
        mpi_time, mpi_acc, mpi_f1 = train_nn_mpi(X_train, y_train, X_test, y_test, n_procs)
        mpi_results.append((f"MPI n_procs={n_procs}", mpi_time, mpi_acc, mpi_f1))
    visualize_results(mpi_results, "mpi_analysis")

    # Poznámka: Testování s rostoucí velikostí dat lze přidat vytvořením větších datasetů
    # Např. změnou velikosti X_train a y_train v load_data()

if __name__ == "__main__":
    main()