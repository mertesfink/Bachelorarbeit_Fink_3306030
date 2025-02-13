import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import PredictionErrorDisplay
import os

def print_results(results):
    for model_name, metrics in results.items():
        print("Modelname:", metrics['Model_name'])
        print("CV Train MAE:", metrics['CV_TrainMAE'])
        print("CV Train RMSE:", metrics['CV_TrainRMSE'])
        print("CV Test MAE:", metrics['CV_TestMAE'])
        print("CV Test RMSE:", metrics['CV_TestRMSE'])
        print("CV Fit Time:", metrics['CV_fit_time'])
        print("-----------------------------------")
        print("Validation MAE:", metrics['ValidationMAE'])
        print("Validation RMSE:", metrics['ValidationRMSE'])
        print("-----------------------------------")
        print("Test MAE:", metrics['TestMAE'])
        print("Test RMSE:", metrics['TestRMSE'])
        print("Train Time:", metrics['TrainTime'])
        print("-----------------------------------")
        print()
        
def model_results_barplot(selection_methode, results,evaluation,name, metrics_upper_limit=None, runtime_upper_limit=None, plot_runtime=True, FS=False):
    
     # Erstelle ein Balkendiagramm
    fig, ax1 = plt.subplots(figsize=(max(len(results) * 1, 10), 10))

    # Mindestbreite für die Balken
    min_bar_width = 0.1

    # Berechne die Balkenbreite basierend auf der Anzahl der Algorithmen
    bar_width = max(min_bar_width, 0.8 / len(results))

    # Positionen für die x-Achsenbeschriftungen
    positions = np.arange(len(results))
    
    label = ""
    mean_absolute_error = []
    root_mean_squared_error = []
    runtime = []
    
    if(evaluation=='cv_train'):
    
        label = "CV_Train"
        # Balken für Mean Absolute Error
        mean_absolute_error = [metrics['CV_TrainMAE'] for metrics in results.values()]
        # Balken für Root Mean Squared Error
        root_mean_squared_error = [metrics['CV_TrainRMSE'] for metrics in results.values()]
        
        if(plot_runtime):
            # Balken für die Laufzeit
            runtime = [metrics['CV_fit_time_ges'] for metrics in results.values()]
                
        
    elif(evaluation=='cv_test'):
        
        label = "CV_Test"
        # Balken für Mean Absolute Error
        mean_absolute_error = [metrics['CV_TestMAE'] for metrics in results.values()]
        # Balken für Root Mean Squared Error
        root_mean_squared_error = [metrics['CV_TestRMSE'] for metrics in results.values()]
        
        if(plot_runtime):
            # Balken für die Laufzeit
            runtime = [metrics['CV_fit_time_ges'] for metrics in results.values()]
            
    elif(evaluation=='train'):
        
        label = "Train"
        # Balken für Mean Absolute Error
        mean_absolute_error = [metrics['TrainMAE'] for metrics in results.values()]
        # Balken für Root Mean Squared Error
        root_mean_squared_error = [metrics['TrainRMSE'] for metrics in results.values()]
        
        if(plot_runtime):
            # Balken für die Laufzeit
            runtime = [metrics['TrainTime_ges'] for metrics in results.values()]
        
    elif(evaluation=='validation'):
        
        label = "Validation"
        # Balken für Mean Absolute Error
        mean_absolute_error = [metrics['ValidationMAE'] for metrics in results.values()]
        # Balken für Root Mean Squared Error
        root_mean_squared_error = [metrics['ValidationRMSE'] for metrics in results.values()]
        
        if(plot_runtime):
            # Balken für die Laufzeit
            runtime = [metrics['TrainTime_ges'] for metrics in results.values()]
            
    elif(evaluation=='test'):
        
        label = "Test"
        # Balken für Mean Absolute Error
        mean_absolute_error = [metrics['TestMAE'] for metrics in results.values()]
        # Balken für Root Mean Squared Error
        root_mean_squared_error = [metrics['TestRMSE'] for metrics in results.values()]
        
        if(plot_runtime):
            # Balken für die Laufzeit
            runtime = [metrics['TrainTime_ges'] for metrics in results.values()]
    else:
        return None
    
    
    ax1.bar(positions - bar_width, mean_absolute_error, color='blue', alpha=0.7, width=bar_width, label='Mean Absolute Error')
    ax1.bar(positions, root_mean_squared_error, color='orange', alpha=0.7, width=bar_width, label='Root Mean Squared Error')
    
    # Beschriftungen und Titel für die erste Achse
    ax1.set_xlabel('Machine Learning Algorithmen')
    ax1.set_ylabel('Metrik Wert')
    ax1.set_title('Vergleich der Algorithmen nach Metriken: '+label)

    # Zeige Gitterlinien für die erste Achse an
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Passe x-Achsenpositionen für bessere Lesbarkeit an
    ax1.set_xticks(positions - bar_width / 2)
    ax1.set_xticklabels(results.keys(), rotation=45, ha='right')

    # Legende für die erste Achse
    ax1.legend(loc='upper left')

    # Plotte Laufzeit, falls plot_runtime=True
    if plot_runtime:
        # Erstelle eine zweite Achse für die Laufzeit
        ax2 = ax1.twinx()

        ax2.bar(positions + bar_width, runtime, color='green', alpha=0.7, width=bar_width, label='Gesamtlaufzeit')

        # Beschriftungen und Titel für die zweite Achse
        ax2.set_ylabel('Laufzeit (s)')
        ax2.legend(loc='upper right')

        # Setze die Y-Achsen-Grenzen für beide Achsen
        if metrics_upper_limit is None:
            metrics_upper_limit = max(max(mean_absolute_error), max(root_mean_squared_error)) * 1.1
        if runtime_upper_limit is None:
            runtime_upper_limit = max(runtime) * 1.1
        ax1.set_ylim(0, metrics_upper_limit)
        ax2.set_ylim(0, runtime_upper_limit)
        
        # Zusätzlicher Balken für die Laufzeit der Feature Selection
        if FS:
            fs_runtime = [metrics['FS-Laufzeit'] for metrics in results.values()]
            ax2.bar(positions + bar_width * 2, fs_runtime, color='red', alpha=0.7, width=bar_width, label='FS-Laufzeit')
        

    else:
        # Setze die Y-Achsen-Grenze nur für die erste Achse
        if metrics_upper_limit is None:
            metrics_upper_limit = max(max(mean_absolute_error), max(root_mean_squared_error)) * 1.1
        ax1.set_ylim(0, metrics_upper_limit)
        
        # Zusätzlicher Balken für die Laufzeit der Feature Selection
        if FS:
            fs_runtime = [metrics['FS-Laufzeit'] for metrics in results.values()]
            ax2.bar(positions + bar_width * 2, fs_runtime, color='red', alpha=0.7, width=bar_width, label='FS-Laufzeit')
        
        

    # Zeige das Diagramm an
    plt.savefig('plots/'+selection_methode+'/vergleich_alg/'+name+"_"+label+'.png')
    plt.show()
    
def predicitons_scatter(selection_methode,predictions, evaluation, name):
    
    if evaluation == 'train':
        y_true = predictions['y_train']
        y_pred = predictions['pred_train']
        
    elif evaluation == 'validation':
        y_true = predictions['y_val']
        y_pred = predictions['pred_val']
        
    elif evaluation == 'test':
        y_true = predictions['y_test']
        y_pred = predictions['pred_test']
        
    else:
        return
    
     # Sicherstellen, dass es ein NumPy-Array ist
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Plot erstellen mit sklearn PredictionErrorDisplay
    fig, ax = plt.subplots(figsize=(6, 6))
    
    PredictionErrorDisplay.from_predictions(
        y_true=y_true, 
        y_pred=y_pred, 
        #kind="residual_vs_predicted",
        kind="actual_vs_predicted",
        subsample=None,
        scatter_kwargs={"s": 5},
        ax=ax
    )

    ax.set_title(f'True vs. Pred: {predictions["Model_name"]} - {evaluation}')

    # Sicherstellen, dass der Ordner existiert, bevor gespeichert wird
    save_path = f'plots/{selection_methode}/predictions/scatterplots/'
    os.makedirs(save_path, exist_ok=True)
    
    plt.savefig(f'{save_path}{name}_{predictions["Model_name"]}_{evaluation}_scatter.png')
    plt.show()
    
    
def residuen_histo(selection_methode,predictions, evaluation, name, range_min=None, range_max=None):
    
    if evaluation == 'train':
        y_true = predictions['y_train']
        y_pred = predictions['pred_train']
        
    elif evaluation == 'validation':
        y_true = predictions['y_val']
        y_pred = predictions['pred_val']
        
    elif evaluation == 'test':
        y_true = predictions['y_test']
        y_pred = predictions['pred_test']
        
    else:
        return
    
    residuen = np.array(y_true) - np.array(y_pred)

    if range_min is None and range_max is None:
        range_min = np.min(residuen)
        range_max = np.max(residuen)
        
    elif range_min is None and range_max is not None:
        range_min = np.min(residuen)
        range_max = range_max
        
    elif range_min is not None and range_max is None:
        range_min = range_min
        range_max = np.max(residuen)
        
    else:
        range_max = range_max
        range_min = range_min
    
    plt.hist(residuen, range=(range_min, range_max), bins=200, color='skyblue', density=True, alpha=0.6, label='Residuen')
    
    # Fit der Residuen auf eine Normalverteilung
    mu, sigma = norm.fit(residuen)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=1, label='Normalverteilung')
  

    plt.title('Residuen: '+predictions['Model_name']+" "+evaluation)
    plt.xlabel('Residuen', fontsize=9, fontweight='bold')
    plt.ylabel('Häufigkeit', fontsize=9, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/'+selection_methode+'/predictions/residuen_histo/' + name + "_" + predictions['Model_name'] + "_" + evaluation +"_residuen"+ '.png')
    plt.show()
    
def plot_metrics_vs_feature_count(df_metrics, models, colors, output_dir, save_path):
    """
    Diese Funktion erstellt und speichert Plots für MAE, RMSE, Laufzeit und Feature Selection Laufzeit
    in Bezug auf die Anzahl der Features.

    Args:
    - df_metrics: DataFrame mit den Metriken, der die Metriken für jedes Modell enthält.
    - models: Liste von Modellnamen.
    - colors: Liste von Farben für jedes Modell.
    - output_dir: Verzeichnis mit den gespeicherten Ergebnissen (CSV-Dateien).
    - save_path: Der Pfad, an dem die Plots gespeichert werden sollen.
    """
    
    # Listen, um die Metriken zu speichern
    mae_values = {model: [] for model in models}
    rmse_values = {model: [] for model in models}
    train_time_values = {model: [] for model in models}  # Liste für Laufzeit
    fs_time_values = {model: [] for model in models}  # Liste für Feature Selection Laufzeit
    feature_counts = [10, 20, 30, 40]

    # Iteriere über jede Feature-Anzahl und extrahiere die Metriken aus den df_metrics
    for num_features in feature_counts:
        # Lese die Metriken für jedes Modell und jedes Feature
        for model in models:
            mae_values[model].append(df_metrics.loc[df_metrics['Model_name'] == model, 'TestMAE'].values[0])
            rmse_values[model].append(df_metrics.loc[df_metrics['Model_name'] == model, 'TestRMSE'].values[0])
            train_time_values[model].append(df_metrics.loc[df_metrics['Model_name'] == model, 'TrainTime_ges'].values[0])  # Laufzeit
            fs_time_values[model].append(df_metrics.loc[df_metrics['Model_name'] == model, 'FS-Laufzeit'].values[0])  # FS Laufzeit

    # Berechne den maximalen Wert für die Metriken MAE, RMSE, Laufzeit und FS Laufzeit, um die y-Achse dynamisch anzupassen
    max_mae = max(max(mae_values[model]) for model in models)
    max_rmse = max(max(rmse_values[model]) for model in models)
    max_train_time = max(max(train_time_values[model]) for model in models)
    max_fs_time = max(max(fs_time_values[model]) for model in models)

    # Plot MAE
    plt.figure(figsize=(10, 6))
    for model, color in zip(models, colors):
        plt.plot(feature_counts, mae_values[model], label=model, color=color, marker='o')

    # Achsen und Titel setzen
    plt.title('MAE (Mean Absolute Error) vs Feature Count')
    plt.xlabel('Feature Count')
    plt.ylabel('MAE')
    plt.xticks(feature_counts)
    plt.ylim(0, max_mae * 1.1)  # Setze die y-Achse bei 0, mit etwas Platz über dem höchsten Wert
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mae_vs_feature_count.png'))
    plt.show()

    # Plot RMSE
    plt.figure(figsize=(10, 6))
    for model, color in zip(models, colors):
        plt.plot(feature_counts, rmse_values[model], label=model, color=color, marker='o')

    # Achsen und Titel setzen
    plt.title('RMSE (Root Mean Squared Error) vs Feature Count')
    plt.xlabel('Feature Count')
    plt.ylabel('RMSE')
    plt.xticks(feature_counts)
    plt.ylim(0, max_rmse * 1.1)  # Setze die y-Achse bei 0, mit etwas Platz über dem höchsten Wert
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'rmse_vs_feature_count.png'))
    plt.show()

    # Plot Laufzeit
    plt.figure(figsize=(10, 6))
    for model, color in zip(models, colors):
        plt.plot(feature_counts, train_time_values[model], label=model, color=color, marker='o')

    # Achsen und Titel setzen
    plt.title('Training Time vs Feature Count')
    plt.xlabel('Feature Count')
    plt.ylabel('Training Time (in seconds)')
    plt.xticks(feature_counts)
    plt.ylim(0, max_train_time * 1.1)  # Setze die y-Achse bei 0, mit etwas Platz über dem höchsten Wert
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'train_time_vs_feature_count.png'))
    plt.show()

    # Plot Feature Selection Laufzeit
    plt.figure(figsize=(10, 6))
    for model, color in zip(models, colors):
        plt.plot(feature_counts, fs_time_values[model], label=model, color=color, marker='o')

    # Achsen und Titel setzen
    plt.title('Feature Selection Time vs Feature Count')
    plt.xlabel('Feature Count')
    plt.ylabel('Feature Selection Time (in seconds)')
    plt.xticks(feature_counts)
    plt.ylim(0, max_fs_time * 1.1)  # Setze die y-Achse bei 0, mit etwas Platz über dem höchsten Wert
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'fs_time_vs_feature_count.png'))
    plt.show()