import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
        
def model_results_barplot(results,evaluation,name, metrics_upper_limit=None, runtime_upper_limit=None, plot_runtime=True, FS=False):
    
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
    plt.savefig('plots/vergleich_alg/'+name+"_"+label+'.png')
    plt.show()
    
def predicitons_scatter(predictions, evaluation, name, alpha=1, s=1, y_true_max=None):
    
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
        
    plt.scatter(y_true, y_pred, alpha=alpha, s=s)

    # Identitätslinie hinzufügen
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='-', alpha=0.4, label='Identität')
    
    # Grenzen für y_true setzen, wenn sie angegeben wurden
    if(y_true_max == None):
        
        lim_max = max(max(y_true), max(y_pred)) + 1
        
    else:
        lim_max = y_true_max
        
    lim_min = min(min(y_true), min(y_pred)) - 1
    
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.title('True vs. Pred: ' +predictions['Model_name']+" ") #+evaluation
    
    
    plt.savefig('plots/predictions/scatterplots/' + name + "_" + predictions['Model_name'] + "_" + evaluation +"_scatter"+ '.png')
    plt.show()
    
def residuen_histo(predictions, evaluation, name, range_min=None, range_max=None):
    
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
    plt.savefig('plots/predictions/residuen_histo/' + name + "_" + predictions['Model_name'] + "_" + evaluation +"_residuen"+ '.png')
    plt.show()