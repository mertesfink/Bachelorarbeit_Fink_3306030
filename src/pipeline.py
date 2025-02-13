from sklearn.base import TransformerMixin
from sklearn.preprocessing import PowerTransformer
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

class DataFrameSubsetTransformer(TransformerMixin):
    
    """
    Extrahiert ein Dataframe-Subset basierend auf den angegebenen Features.
    """

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
     
        return self

    def transform(self, X):
        
        subset = X[self.features]  # Extrahiere das Subset des DataFrames basierend auf den Features
        return subset
    
    
class RemoveColumns(TransformerMixin):
    """
    Entfernt eine Liste von Spalten aus einem DataFrame.
    """

    def __init__(self, column_names):
        # Setze die Liste der Spaltennamen, die entfernt werden sollen
        if isinstance(column_names, list):
            self.column_names = column_names
        else:
            raise ValueError("column_names sollte eine Liste von Spaltennamen sein.")
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Kopie des DataFrames erstellen, um das Original nicht zu ändern
        X_removed = X.drop(columns=self.column_names, errors='ignore')
        return X_removed

class DropNaNsInColumn(TransformerMixin):
    
    """
    Droppt in der angegebenen Spalte NaN-Werte und gibt das bereinigte Dataframe zurück.
    """

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):

        return self

    def transform(self, X):
      
        # Kopie des DataFrames erstellen, um das Original nicht zu ändern
        X_cleaned = X.copy()

        # Anzahl der NaN-Werte vor dem Entfernen zählen
        nans_before = X_cleaned[self.column_name].isna().sum()

        # NaN-Werte in der angegebenen Spalte entfernen
        X_cleaned = X_cleaned.dropna(subset=[self.column_name])

        # Anzahl der NaN-Werte nach dem Entfernen zählen
        nans_after = X_cleaned[self.column_name].isna().sum()

        # Berechne die Anzahl der entfernten NaN-Werte und drucke sie
        dropped_count = nans_before - nans_after
        print("Anzahl der entfernten NaN-Werte in Spalte '{}': {}".format(self.column_name, dropped_count))

        return X_cleaned
    
class CleanFeatureNamesTransformer(TransformerMixin):
    
    """
    Entfernt problematische Zeichen in den Spaltennamen und ersetzt diese durch Unterstriche.
    """
    

    def fit(self, X, y=None):
        
        return self

    def transform(self, X):

        # Kopie des DataFrame erstellen, um die Originaldaten nicht zu verändern
        X_cleaned = X.copy()

        # Iteriere über die Spaltennamen und ersetze problematische Zeichen durch Unterstriche
        for col in X_cleaned.columns:
            col_cleaned = col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            # Überprüfe, ob sich der Name geändert hat, und aktualisiere ihn
            if col != col_cleaned:
                X_cleaned.rename(columns={col: col_cleaned}, inplace=True)

        return X_cleaned

class SameValuesColumnRemover(TransformerMixin):
    
    """
    Entfernt Spalten mit konstantem Wert und mit einem gewissen Prozentsatz an NaN-Werten in einem Dataframe und gibt das bereinigte Dataframe zurück.
    """

    def __init__(self, nan_threshold=0):
        self.nan_threshold = nan_threshold
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        constant_columns = []
        
        # Überprüfen auf konstante Werte in jeder Spalte, einschließlich NaN
        for col in X.columns:
            
            unique_values = X[col].nunique()
            nan_count = X[col].isna().sum()
            nan_ratio = nan_count / len(X)
            
            if unique_values == 1 or nan_ratio> self.nan_threshold:
                constant_columns.append(col)
        
        # Entfernen der konstanten Spalten aus dem DataFrame
        X_cleaned = X.drop(columns=constant_columns)
        
        return X_cleaned
    
class SimilarityCalculator(TransformerMixin):
    
    """
    Berechnet die Übereinstimmung von zwei angegebenen Spalten in einem Dataframe.
    """
    
    def __init__(self, col1, col2):
        self.col1 = col1
        self.col2 = col2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Berechne die absolute Differenz zwischen den Werten in den Spalten
        absolute_diff = abs(X[self.col1] - X[self.col2])
        # Berechne die durchschnittliche Übereinstimmung der Werte
        avg_similarity = (1 - absolute_diff / (X[self.col1].max() - X[self.col1].min())) * 100
        avg_similarity = avg_similarity.mean()
        print(f"Durchschnittliche Übereinstimmung zwischen {self.col1} und {self.col2}: {avg_similarity}%")
        return X


class ImputerTransformer(TransformerMixin):
    
    """
    Imputiert die NaN-Werte mithilfe einer angegebenen Methode.
    """
    
    def __init__(self, strategy='median'):
        self.strategy = strategy
    
    def fit(self, X, y=None):
        # Zähle NaN-Werte vor der Imputation
        self.nan_count_before = np.isnan(X).sum().sum()
        
        # Erzeuge einen SimpleImputer mit der gewählten Strategie
        self.imputer = SimpleImputer(strategy=self.strategy)
        
        # Führe die Imputation auf den Daten durch
        start_time = time.time()
        self.imputer.fit(X)
        end_time = time.time()
        self.fit_time = end_time - start_time
        
        return self
    
    def transform(self, X):
        # Wende die Imputation auf die Daten an
        start_time = time.time()
        X_imputed = self.imputer.transform(X)
        end_time = time.time()
        transformation_time = end_time - start_time
        
        print("Imputation : Laufzeit der Imputation: {:.4f} Sekunden".format(transformation_time+self.fit_time))
        print("------------------------")
        
        return pd.DataFrame(X_imputed, columns=X.columns)
    
class PowerTransformerCustom(TransformerMixin):
    
    """
    Führt eine Yeo-Johnson-Transformation auf den Daten durch.
    """

    def __init__(self, method='yeo-johnson'):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        start_time = time.time()
        
        pt = PowerTransformer(method=self.method)
        transformed_data = pt.fit_transform(X)
        transformed_X = pd.DataFrame(transformed_data, columns=X.columns)

        end_time = time.time()
        runtime = end_time - start_time
        print("Transformation: Laufzeit der Yeo-Johnson-Transformation: {:.4f} Sekunden".format(runtime))
        print("------------------------")

        return pd.DataFrame(transformed_X, columns=X.columns)

class RobustScalerTransformer(TransformerMixin):
    
    """
    Führt eine Robust-Skalierung auf den Daten durch.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        # Erzeuge einen RobustScaler
        self.scaler = RobustScaler()
        
        # Wende die Skalierung auf die Daten an und messe die Laufzeit
        start_time = time.time()
        X_scaled = self.scaler.fit_transform(X)
        end_time = time.time()
        transformation_time = end_time - start_time
        
        # Drucke die Laufzeit der Skalierung und Transformation
        print("Skalierung: Laufzeit der Skalierung: {:.4f} Sekunden".format(transformation_time))
        print("------------------------")
        return pd.DataFrame(X_scaled, columns=X.columns)

class IQRTransformer(TransformerMixin):
    
    """
    Führt die Ausreißerentfernung der IQR-Methode durch.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Nichts zu trainieren, daher Rückgabewert ist self
        return self
    
    def transform(self, X):
        start_time = time.time()

        total_points = X.size  # Gesamtanzahl der Datenpunkte

        changed_points = 0  # Initialisierung für die Anzahl der veränderten Datenpunkte

        X_transformed = X.copy()  # Kopie des DataFrames erstellen, um das Original nicht zu verändern

        for feature in X_transformed.columns:
            
            sorts = X_transformed[feature].sort_values()
            
            iqr = sorts.quantile(0.75) - X_transformed[feature].quantile(0.25)
            lower = sorts.quantile(0.25) - iqr * 1.5
            upper = sorts.quantile(0.75) + iqr * 1.5
            clipped_values = X_transformed[feature].clip(lower, upper)

            # Zähle die Anzahl der veränderten Datenpunkte
            changed_points += clipped_values[clipped_values != X_transformed[feature]].count()

            X_transformed[feature] = clipped_values

        end_time = time.time()
        runtime = end_time - start_time

        # Berechne den Prozentsatz der veränderten Datenpunkte
        percentage_changed = (changed_points / total_points) * 100

        # Drucke die Laufzeit und den Prozentsatz der veränderten Datenpunkte
        print("IQR: Laufzeit der IQR-Berechnung: {:.4f} Sekunden".format(runtime))
        print("IQR: Prozentsatz der veränderten Datenpunkte: {:.2f}%".format(percentage_changed))
        print("------------------------")

        return pd.DataFrame(X_transformed, columns=X.columns)
    
class MissingRowsCounter(TransformerMixin):
    
    """
    Berechnet den Prozentsatz der Zeilen, welcher ohne Imputation gedroppt würde.
    """
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        # Zähle die Anzahl der Zeilen mit mindestens einem fehlenden Wert
        rows_with_missing_values = X.isna().any(axis=1).sum()
        
        # Berechne den Prozentsatz der Zeilen, die gedroppt würden
        total_rows = X.shape[0]
        percentage_rows_dropped = (rows_with_missing_values / total_rows) * 100
        
        print("Prozentsatz der Zeilen, die gedroppt würden, wenn keine Imputation stattfindet:", percentage_rows_dropped, "%")
        
        return X

class CategoricalModeImputer(TransformerMixin):
    
    """
    Füllt die kategorischen Werte eines Dataframes mit dem Modus der jeweiligen Spalte.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        modes = X.mode().iloc[0]
        
        X_filled = X.copy()
        total_rows = len(X_filled)
        total_missing = X_filled.isna().sum().sum()  # Gesamtzahl der fehlenden Werte in allen kategorischen Spalten
        
        # Auffüllen der fehlenden Werte in kategorischen Spalten mit ihrem Modus
        for column in X_filled.columns:
            filled_values = X_filled[column].fillna(modes.get(column, None))
            X_filled[column] = filled_values
        
        return pd.DataFrame(X_filled, columns=X.columns)
    
class HybridEncoder(TransformerMixin):
    """
    Hybride Enkodierung kategorialer Daten basierend auf ihrer Kardinalität,
    ohne One-Hot-Encoding.
    """

    def __init__(self, high_cardinality_threshold=50):
        """
        Initialisiert den Encoder mit Parametern für Kardinalitätsschwellenwerte.

        Args:
            high_cardinality_threshold (int): Schwellenwert für hohe Kardinalität.
        """
        self.high_cardinality_threshold = high_cardinality_threshold
        self.encoders_ = {}
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Transformiert die Daten basierend auf der Kardinalität jeder Spalte.

        Args:
            X (pd.DataFrame): Eingabedaten mit kategorialen Spalten.
            y (pd.Series, optional): Zielwerte, notwendig nur für Target-Encoding.

        Returns:
            pd.DataFrame: Der transformierte DataFrame.
        """
        X_transformed = X.copy()
        encoded_features = 0

        for column in X.columns:
            unique_vals = X[column].nunique()

            if unique_vals <= self.high_cardinality_threshold:
                # Niedrige Kardinalität → Frequency-Encoding
                freq_encoding = X[column].value_counts() / len(X)
                X_transformed[column] = X[column].map(freq_encoding)
                
            elif unique_vals > self.high_cardinality_threshold:
                # Hohe Kardinalität → Target-Encoding
                encoder = TargetEncoder(target_type="continuous")
                X_transformed[column] = encoder.fit_transform(X[[column]], y)

            encoded_features += 1

        print(f"{encoded_features} Merkmale wurden encodet.")
        return X_transformed
    
class DateTimePreprocessor(TransformerMixin):
    
    """
    Preprocesst die Datum-Merkmale eines Dataframes.
    """

    def __init__(self, fill_method='ffill'):
        self.fill_method = fill_method
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        X_cleaned = X.copy()
        datetime_columns = X.columns
        
        for col in datetime_columns:
            X_cleaned[col] = pd.to_datetime(X_cleaned[col], errors='coerce')
            
            X_cleaned[col] = X_cleaned[col].fillna(method=self.fill_method)
            if X_cleaned[col].isna().values.any():
                X_cleaned[col] = X_cleaned[col].fillna(method='bfill')
               
            # Generate time dummy: count steps from the beginning
            earliest_date = X_cleaned[col].min()
            X_cleaned[col] = (X_cleaned[col] - earliest_date).dt.days
   
        return X_cleaned

    
def einteilen_merkmale(df):
    
    """
    Teilt das Dataframe in die unterschiedlichen Datentypen ein.
    """
    
    numerische_merkmale = []
    kategorische_merkmale = []
    restliche_merkmale = []

    for spalte, datentyp in df.dtypes.items():
        if pd.api.types.is_numeric_dtype(datentyp):
            numerische_merkmale.append(spalte)
        elif pd.api.types.is_categorical_dtype(datentyp) or pd.api.types.is_string_dtype(datentyp):
            kategorische_merkmale.append(spalte)
        else:
            restliche_merkmale.append(spalte)

    return numerische_merkmale, kategorische_merkmale, restliche_merkmale