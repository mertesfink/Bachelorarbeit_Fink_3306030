import pandas as pd
import time
import numpy as np
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def correlation(X, y, threshold=0.5):
    """
    Führt eine Feature Selection basierend auf dem Korrelationskoeffizienten mit der Zielvariable durch.

    Parameters:
    - X (np.ndarray): Die Features als numpy-Array.
    - y (np.ndarray): Die Zielvariablen als numpy-Array.
    - threshold (float): Der Schwellenwert für die Korrelation, unterhalb dessen Features entfernt werden.

    Returns:
    - selected_features (list): Eine Liste der ausgewählten Features.
    - num_selected_features (int): Die Anzahl der ausgewählten Features.
    - duration (float): Die Dauer der Feature Selection in Sekunden.
    """
    start_time = time.time()  # Startzeit der Feature Selection
    
    # Korrelationen zwischen Features und Zielvariablen berechnen
    correlations = np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    
    # Ausgewählte Features mit Korrelation über dem Schwellenwert extrahieren
    selected_features = np.where(np.abs(correlations) >= threshold)[0]
    
    # Anzahl der ausgewählten Features
    num_selected_features = len(selected_features)
    
    end_time = time.time()  # Endzeit der Feature Selection
    duration = end_time - start_time  # Dauer der Feature Selection
    
    return selected_features, num_selected_features, duration

def rfe(model, X, y, n_features_to_select, step=1):
    """
    Führt eine Feature Selection mit Recursive Feature Elimination (RFE) für ein einzelnes Modell durch.

    Parameters:
    - model: Das Modell, das für die Feature Selection verwendet werden soll.
    - X (pd.DataFrame oder np.array): Die Feature-Daten.
    - y (pd.Series oder np.array): Die Zielvariablen.
    - n_features_to_select (int): Die Anzahl der Features, die ausgewählt werden sollen.

    Returns:
    - selected_features (list): Eine Liste der ausgewählten Feature-Indizes.
    - duration (float): Die Dauer der Feature Selection in Sekunden.
    """
    start_time = time.time()  # Startzeit der Feature Selection
    
    # Initialisiere RFE mit dem aktuellen Modell und der Anzahl der gewünschten Features
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select, step=step)
    # Führe RFE auf den Daten durch
    rfe.fit(X, y)
    # Extrahiere die ausgewählten Features
    selected_features = rfe.support_
    
    # Anzahl der ausgewählten Features
    num_selected_features = len(selected_features)
    
    end_time = time.time()  # Endzeit der Feature Selection
    duration = end_time - start_time  # Dauer der Feature Selection

    return selected_features, num_selected_features, duration

def visualize_rfe_runtime(model, X, y, max_features, step=1, time_limit=600):
    """
    Visualisiert die Laufzeit der schrittweisen Feature Selection mit Recursive Feature Elimination (RFE).

    Parameters:
    - model: Das Modell, für das die Feature Selection durchgeführt werden soll.
    - X (pd.DataFrame oder np.array): Die Feature-Daten.
    - y (pd.Series oder np.array): Die Zielvariablen.
    - max_features (int): Die maximale Anzahl von Features, die ausgewählt werden sollen.
    - step (int): Die Schrittweite für die Reduzierung der ausgewählten Features.
    - time_limit (int): Das Zeitlimit in Sekunden für die Feature Selection.
    """
    start_time = time.time()  # Startzeit der Feature Selection
    
    n_features_to_select = max_features  # Anfangswert für die Anzahl der auszuwählenden Features
    runtime_data = []  # Liste für die Laufzeiten
    n_features_to_remove = 0
    
    # Schleife für die schrittweise Reduzierung der ausgewählten Features
    while n_features_to_remove <= max_features:
        # Startzeit der aktuellen Iteration
        iteration_start_time = time.time()
        
        # Initialisiere RFE mit dem aktuellen Modell und der Anzahl der auszuwählenden Features
        rfe = RFE(estimator=model, n_features_to_select=n_features_to_select, step=1)
        # Führe RFE auf den Daten durch
        rfe.fit(X, y)
        
        # Endzeit der aktuellen Iteration
        iteration_end_time = time.time()
        # Dauer der aktuellen Iteration
        iteration_duration = iteration_end_time - iteration_start_time
        
        # Überprüfe das Zeitlimit für die Feature Selection
        if iteration_duration > time_limit:
            print(f"Zeitlimit von {time_limit} Sekunden überschritten. Feature Selection abgebrochen.")
            break
        
        # Füge die Laufzeit zur Liste hinzu
        runtime_data.append((n_features_to_remove, iteration_duration))
        
        # Reduziere die Anzahl der zu entfernenden Features um die Schrittweite
        n_features_to_remove += step
        n_features_to_select = max_features - n_features_to_remove

    # Extrahiere die Anzahl der zu entfernenden Features und die Laufzeiten
    features_removed, runtimes = zip(*runtime_data)
    
    # Plot der Laufzeit in Abhängigkeit von der Anzahl der zu entfernenden Features
    plt.plot(features_removed, runtimes, marker='o')
    plt.xlabel('Anzahl der zu entfernenden Features')
    plt.ylabel('Laufzeit (Sekunden)')
    plt.title('Laufzeit der Feature Selection mit RFE')
    plt.grid(True)
    plt.show()