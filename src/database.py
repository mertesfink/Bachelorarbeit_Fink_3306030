import duckdb
import pandas as pd
def database_important_features():
    """
    Stellt eine Verbindung zu einer Datenbank her, ruft die Tabelle 'Heimbach_S1_wichtige_Parameter' ab und gibt sie als DataFrame zurück.

    Returns:
    - df (pandas.DataFrame): Ein DataFrame, das die Daten aus der Tabelle 'Heimbach_S1_wichtige_Parameter' enthält.
    """
    try:
        # Verbindung zur Datenbank herstellen
        con = duckdb.connect(database="data/praxisprojekt_heimbach")

        # Daten aus der Tabelle abrufen und in ein DataFrame konvertieren
        df = con.table('Heimbach_S1_wichtige_Paramter').df()

        #Verbindung schließen
        con.close()

        return df
    except Exception as e:
        print(f"Fehler bei der Datenbankverbindung: {str(e)}")
        return pd.DataFrame()  # Leeres DataFrame zurückgeben, wenn ein Fehler auftritt
def database_raw():
    """
    Stellt eine Verbindung zu einer Datenbank her, ruft die Tabelle 'S1_InnerJoin_Labordaten' ab und gibt sie als DataFrame zurück.

    Returns:
    - df (pandas.DataFrame): Ein DataFrame, dass die Daten aus der Tabelle 'S1_InnerJoin_Labordaten' enthält.
    """
    try:
        # Verbindung zur Datenbank herstellen
        con = duckdb.connect(database="data/praxisprojekt_heimbach")

        # Daten aus der Tabelle abrufen und in ein DataFrame konvertieren
        df = con.table('S1_InnerJoin_Labordaten').df()

        # Verbindung schließen
        con.close() 

        return df
    except Exception as e:
        print(f"Fehler bei der Datenbankverbindung: {str(e)}")
        return pd.DataFrame()  # Leeres DataFrame zurückgeben, wenn ein Fehler auftritt 
    
def get_heimbach_important_features():
    """
    Stellt eine Verbindung zur Datenbank her, lädt die von den Domänexperten augesuchten Merkmale und ergänzt FELT_LIFE_NET.

    Returns:
    - important_features (list): Eine Liste von Merkmalsnamen
    """
    # Rufe die wichtigen Merkmale aus der Datenbank ab
    important_features_from_database = database_important_features()

    # Extrahiere die Spalte 'parameter_de' aus den wichtigen Merkmalen
    important_features = important_features_from_database['parameter_de'].tolist()

    # Füge das Label 'FELT_LIFE_NET' hinzu
    important_features.append('FELT_LIFE_NET')

    return important_features