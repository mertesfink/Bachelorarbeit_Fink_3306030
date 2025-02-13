import pandas as pd
import time
import numpy as np
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


def correlation(X, y, num_features):
    """
    Führt eine Feature Selection basierend auf dem Korrelationskoeffizienten mit der Zielvariable durch.

    Parameters:
    - X (np.ndarray): Die Features als numpy-Array.
    - y (np.ndarray): Die Zielvariablen als numpy-Array.
    - num_features (int): Anzahl der auszuwählenden Features.

    Returns:
    - selected_features (list): Eine Liste der ausgewählten Features.
    - duration (float): Die Dauer der Feature Selection in Sekunden.
    """
    start_time = time.time()  # Startzeit der Feature Selection
    
    # Korrelationen zwischen Features und Zielvariablen berechnen
    correlations = np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    
    # Features nach absoluter Korrelation sortieren (höchste zuerst)
    sorted_indices = np.argsort(np.abs(correlations))[::-1]
    
    # Die ersten num_features auswählen
    selected_features = sorted_indices[:num_features]
    
    end_time = time.time()  # Endzeit der Feature Selection
    duration = end_time - start_time  # Dauer der Feature Selection
    
    return selected_features, duration

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
    
    n_features_to_select = step  # Anfangswert für die Anzahl der auszuwählenden Features
    runtime_data = []  # Liste für die Laufzeiten
    runtime_data.append((0, 0))
    
    # Schleife für die schrittweise Reduzierung der ausgewählten Features
    while n_features_to_select <= max_features:
        # Startzeit der aktuellen Iteration
        iteration_start_time = time.time()
        
        print(n_features_to_select)
        
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
        runtime_data.append((n_features_to_select, iteration_duration))
        
        # Reduziere die Anzahl der zu entfernenden Features um die Schrittweite
        n_features_to_select += step

    # Extrahiere die Anzahl der zu entfernenden Features und die Laufzeiten
    features_removed, runtimes = zip(*runtime_data)
    
    # Plot der Laufzeit in Abhängigkeit von der Anzahl der zu entfernenden Features
    plt.plot(features_removed, runtimes, marker='o')
    plt.xlabel('Anzahl der auszuwählenden Features')
    plt.ylabel('Laufzeit (Sekunden)')
    plt.title('Laufzeit der Feature Selection mit RFE')
    plt.grid(True)
    plt.show()

def pca(X_train, X_test, num_features):
    """
    Führt PCA durch: Fit auf Trainingsdaten, Transform auf allen Datensets.
    
    Args:
    - x_train (numpy array): Trainingsdaten.
    - x_test (numpy array): Testdaten.
    - n_components (float or int): Anzahl der Hauptkomponenten 
    
    Returns:
    - x_train_pca (numpy array): Transformierte Trainingsdaten.
    - x_test_pca (numpy array): Transformierte Testdaten.
    - duration (float): Zeit, die für die PCA benötigt wurde.
    - pca (PCA object): Das gefittete PCA-Objekt (optional, falls du die PCA-Parameter analysieren willst).
    """
    
    start_time = time.time()  # Startzeit der PCA
    
    # PCA-Objekt erstellen und anpassen
    pca = PCA(n_components=num_features)
    x_train_pca = pca.fit_transform(X_train)
    
    # Transformiere die Testdaten
    x_test_pca = pca.transform(X_test)
    
    end_time = time.time()  # Endzeit der PCA
    duration = end_time - start_time  # Dauer der PCA-Feature-Selection

    return x_train_pca, x_test_pca, duration, pca

def rf_feature_selection(X_train, y_train, num_features):
    """
    Führt eine Feature Selection mithilfe eines RandomForestRegressors durch.
    
    Args:
    - X_train (numpy array or pandas DataFrame): Trainingsdaten.
    - y_train (numpy array): Zielvariable.
    - num_features (int): Anzahl der besten Features, die extrahiert werden sollen.
    
    Returns:
    - selected_features (list): Die `num_features` wichtigsten Features.
    - feature_importances (numpy array): Die Importanzen der Features.
    - duration (float): Die Dauer der Berechnung in Sekunden.
    """
    
    start_time = time.time()
    
    #Feature Importance Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Feature Importance extrahieren
    feature_importances = rf_model.feature_importances_
    
    # Indizes der besten Features basierend auf der Importance
    selected_features = np.argsort(feature_importances)[::-1][:num_features]
    
    end_time = time.time() 
    duration = end_time - start_time
        
    return selected_features, feature_importances[selected_features], duration