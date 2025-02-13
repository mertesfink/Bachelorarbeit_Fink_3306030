"""
Code basierend auf dem GitHub-Repository von Ghaith81 zum 
Repository: https://github.com/Ghaith81/Fast-Genetic-Algorithm-For-Feature-Selection

"""
import pandas as pd
import numpy as np
import copy
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import PowerTransformer

import time
import shap

class Dataset:

   # The init method or constructor
    def __init__(self, file_path, sep, label, divide_dataset=True, header=None):
        # Read csv data file
        if (sep == ','):
            df = pd.read_csv(file_path, header=header, sep=',')
        if (sep == ';'):
            df = pd.read_csv(file_path, header=header, sep=';')
        if (sep == ' '):
            df = pd.read_csv(file_path, delim_whitespace=True, header=header)
        if (sep == 'df'):
            df = file_path
            file_path = 'dummy'

        # Set raw data attribute
        self.df = df
        self.label = label
        self.df_sampled = self.df

        if (divide_dataset):
            self.divideDataset()

    def divide_dataset(self, classifier, normalize=True, shuffle=True, all_features=True, all_instances=True,
                      evaluate=True, partial_sample=False, folds=5):

        # Set classifier
        self.clf = copy.copy(classifier)
        self.folds= folds

        # Shuffle dataset
        if (shuffle):
            self.df = self.df.sample(frac=1, random_state=42)
            self.df_sampled = self.df



        # Divide datset into training/validation/testing
        if (self.label == -1):
            self.X = self.df_sampled[:, :-1].values
            self.y = self.df_sampled.iloc[:, -1].values
        else:
            selector = [x for x in range(self.df.shape[1]) if x != self.label]
            self.X = self.df_sampled.iloc[:, selector].values
            self.y = self.df_sampled.iloc[:, self.label].values
        
        
        
        X_train = self.X[:int(0.6 * len(self.X)), :]
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 0.0001
        X_train_normalized = (X_train - mean) / std
        y_train = self.y[:int(0.6 * len(self.X))]

        X_val = self.X[int(0.6 * len(self.X)):int(0.8 * len(self.X)), :]
        y_val = self.y[int(0.6 * len(self.X)):int(0.8 * len(self.X))]
        X_val_normalized = (X_val - mean) / std

        X_test = self.X[int(0.8 * len(self.X)):, :]
        y_test = self.y[int(0.8 * len(self.X)):]
        X_test_normalized = (X_test - mean) / std
        
        self.pt = PowerTransformer(method='yeo-johnson')

        # Set attribute values
        if (normalize):
            self.X_train = X_train_normalized
            self.y_train = y_train
            self.y_train_trans = self.pt.fit_transform(y_train.reshape(-1,1)).flatten()
            self.X_val = X_val_normalized
            self.y_val = y_val
            self.X_test = X_test_normalized
            self.y_test= y_test
            
        else:
            self.X_train = X_train
            self.y_train = y_train
            self.y_train_trans = self.pt.fit_transform(y_train.reshape(-1,1)).flatten()
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test

        # Confirm instances/features to be used for learning
        if (all_features):
            self.features = np.ones(X_train.shape[1])
        else:
            self.features = np.zeros(X_train.shape[1])
            while (np.sum(self.features) == 0):
                zero_p = random.uniform(0, 1)
                # zero_p = 0.5
                self.features = np.random.choice([0, 1], size=(X_train.shape[1],), p=[zero_p, (1 - zero_p)])
        self.features = list(np.where(self.features == 1)[0])

        if (all_instances):
            self.instances = np.ones(X_train.shape[0])
        else:
            self.instances = np.random.choice([0, 1], size=(X_train.shape[0],), p=[0.5, 0.5])
        self.instances = list(np.where(self.instances == 1)[0])

        # Sample from training split
        if (partial_sample):
            self.instances = np.random.choice(self.X_train.shape[0], partial_sample, replace=False)

        # Train model and evaluate on validation/testing sets
        if (evaluate):
            self.fit_classifier()
            self.set_validation_accuracy()
            self.set_test_accuracy()
    
            self.set_train_metrics()
            self.set_validation_metrics()
            self.set_test_metrics()

    def fit_classifier(self):
        # Zeit vor dem Training aufnehmen
        start_time = time.time()
        
        self.clf = self.clf.fit(self.X_train[self.instances,:][:, self.features], self.y_train_trans[self.instances])
        
        # Zeit nach dem Training aufnehmen
        end_time = time.time()
        # Trainingszeit berechnen
        self.clf.training_time = end_time - start_time
    
    def set_train_metrics(self):
        self.y_pred_train = self.pt.inverse_transform(self.clf.predict(self.X_train[:, self.features]).reshape(-1,1)).flatten() 
        
        result_dict = {
        'TrainRMSE': np.sqrt(MSE(self.y_train, self.y_pred_train)),
        'TrainMAE': MAE(self.y_train, self.y_pred_train)
        }
        self.TrainMetrics = result_dict
        
    def set_validation_metrics(self):
        self.y_pred_val = self.pt.inverse_transform(self.clf.predict(self.X_val[:, self.features]).reshape(-1,1)).flatten() 
        
        result_dict = {
        'ValidationRMSE': np.sqrt(MSE(self.y_val, self.y_pred_val)),
        'ValidationMAE': MAE(self.y_val, self.y_pred_val)
        }
        self.ValidationMetrics = result_dict
        
    def set_test_metrics(self):
        self.y_pred_test = self.pt.inverse_transform(self.clf.predict(self.X_test[:, self.features]).reshape(-1,1)).flatten() 
        
        result_dict = {
        'TestRMSE': np.sqrt(MSE(self.y_test, self.y_pred_test)),
        'TestMAE': MAE(self.y_test, self.y_pred_test)
        }
        self.TestMetrics = result_dict

    def set_validation_accuracy(self):
        y_pred = self.pt.inverse_transform(self.clf.predict(self.X_val[:, self.features]).reshape(-1,1)).flatten()
            
        self.ValidationAccuracy =  np.sqrt(MSE(self.y_val, y_pred))

    def set_test_accuracy(self):
        y_pred = self.pt.inverse_transform(self.clf.predict(self.X_test[:, self.features]).reshape(-1,1)).flatten()
        
        self.TestAccuracy = np.sqrt(MSE(self.y_test, y_pred))

    def set_CV(self):
        
        kf = KFold(self.folds)

        # Manuelle Cross-Validation, um Vorhersagen zu erhalten und rückzutransformieren
        results = {}
        
        CV_TrainMAE = []
        CV_TestMAE = []
        CV_TrainRMSE = []
        CV_TestRMSE = []
        CV_fit_time = []
        
        for train_index, test_index in kf.split(self.X_train[:][:, self.features]):
            X_train, X_test = self.X_train[train_index][:, self.features], self.X_train[test_index][:, self.features]
            y_train, y_test = self.y_train_trans[train_index][:], self.y_train_trans[test_index][:]
            
            start_time = time.time()
    
            model = copy.copy(self.clf)
            model.fit(X_train, y_train)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            y_pred_train = self.pt.inverse_transform(model.predict(X_train).reshape(-1,1)).flatten()
            y_pred_test = self.pt.inverse_transform(model.predict(X_test).reshape(-1,1)).flatten()
            
            CV_TrainMAE.append(MAE(self.pt.inverse_transform(y_train.reshape(-1,1)), y_pred_train).flatten())
            CV_TestMAE.append(MAE(self.pt.inverse_transform(y_test.reshape(-1,1)), y_pred_test).flatten())
            CV_TrainRMSE.append(np.sqrt(MSE(self.pt.inverse_transform(y_train.reshape(-1,1)), y_pred_train).flatten()))
            CV_TestRMSE.append(np.sqrt(MSE(self.pt.inverse_transform(y_test.reshape(-1,1)), y_pred_test).flatten()))
            CV_fit_time.append(training_time)
            
        result_dict = {
        'CV_TrainMAE': np.mean(CV_TrainMAE),
        'CV_TrainRMSE': np.mean(CV_TrainRMSE),
        'CV_TestMAE': np.mean(CV_TestMAE),
        'CV_TestRMSE': np.mean(CV_TestRMSE),
        'CV_fit_time': np.mean(CV_fit_time)
        }
        self.CV = result_dict
        
    def plot_shapley_values(self, title):
    
        start_time = time.time()
        
        selected_feature_names = self.df.columns[self.features].tolist()
        
        background_summary = shap.utils.sample(self.X_train[:][:, self.features], nsamples=100, random_state=42)
        
        explainer = shap.KernelExplainer(self.clf.predict, background_summary)
        shapley_values = explainer.shap_values(background_summary)
        
        end_time = time.time()
        shapley_time = end_time - start_time
        
        plt.title("Shapley Summary Plot: "+title+" Berechnungszeit: "+str(round(shapley_time,2)))
        
        shap.summary_plot(shapley_values, features=background_summary,plot_type='bar', feature_names=selected_feature_names)

    def set_train_set(self, selected_instances):
        self.X_train = self.X_train[selected_instances]
        self.y_train_trans = self.y_train_trans[selected_instances]

    def set_features(self, selected_features):
        self.features = selected_features

    def set_instances(self, selected_instances):
        self.instances = selected_instances

    def get_validation_accuracy(self):
        return self.ValidationAccuracy

    def get_test_accuracy(self):
        return self.TestAccuracy

    def get_CV(self):
        return self.CV
    
    def get_train_metrics(self):
        return self.TrainMetrics
    
    def get_validation_metrics(self):
        return self.ValidationMetrics

    def get_test_metrics(self):
        return self.TestMetrics
    
    def get_traintime(self):
        return self.clf.training_time
    
    def get_X_train(self):
        return self.X_train
    
    def get_X_val(self):
        return self.X_val
    
    def get_X_test(self):
        return self.X_test
    
    def get_y_pred_train(self):
        return self.y_pred_train
    
    def get_y_pred_val(self):
        return self.y_pred_val
    
    def get_y_pred_test(self):
        return self.y_pred_test
    
    def get_y_train(self):
        return self.y_train
    
    def get_y_val(self):
        return self.y_val
    
    def get_y_test(self):
        return self.y_test