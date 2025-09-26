# %%
import pandas as pd
from pathlib import Path

def normalize_dataframe(df):
    """
    Cette fonction normalise les colonnes numériques d'une DataFrame entre 0 et 1
    Entrée :
    - df : pandas.DataFrame contenant nos données
    Sortie :
    - pandas.DataFrame avec les colonnes numériques normalisées
    """

    df_normalized = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns # colonnes numériques

    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val != max_val:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0.0  # Valeur constante → mise à 0 (Ca ne devrait pas arriver avec nos données.)

    return df_normalized


def prep_catv(df, colonne="catv"):
    """
    Trier la DataFrame selon la taille réelle des véhicules de la colonne catv.
    Supprime les zéros avant les chiffres, convertit en entier, puis trie selon un ordre logique croissant choisi dans cette fonction.

    Entrée :
    - df : pandas.DataFrame avec la colonne catv
    - colonne : nom de la colonne catv
    Sortie :
    - df trié selon la taille des véhicules
    """
    # Suppression des zéros avant les chiffres et conversion en entier
    df[colonne] = df[colonne].astype(str).str.lstrip("0").replace("", "0").astype(int) # on convertit en string, puis on supprime les zéros de gauche, on remplace les chaînes vides par "0", enfin on re-convertit en entier

    # Ordre logique choisi des véhicules du plus léger au plus lourd
    ordre_logique = [
        60, 50, 80,          # EDP sans moteur, à moteur, VAE
        1,                   # Bicyclette
        2, 30,               # Cyclomoteur, scooter < 50
        31, 32, 41, 42,      # 2RM/3RM <= 125 cm3
        3, 35, 36,           # Voiturette, quads
        33, 34, 43,          # >125 cm3
        7,                   # VL seul
        10,                  # VU 1,5–3,5T
        13, 14, 15,          # PL
        16, 17,              # Tracteur
        21,                  # Tracteur agricole
        20,                  # Engin spécial
        37, 38,              # Autobus, autocar
        39, 40,              # Train, tramway
        0, 4, 5, 6, 8, 9, 11, 12, 18, 19, 99  # Autres ou obsolètes
    ]

    # Créer un dictionnaire de priorité
    ordre_dict = {code: i for i, code in enumerate(ordre_logique)}

    # Ajouter une colonne temporaire de tri
    df["_ordre_vehicule"] = df[colonne].map(ordre_dict) #map() parcourt les valeurs de df[colonne] en les remplaçant par leur correspondance dans ordre_dict.

    # Trier et supprimer la colonne temporaire
    df = df.sort_values(by="_ordre_vehicule").drop(columns="_ordre_vehicule").reset_index(drop=True) # on classe les véhicules du + léger au + lourd,
    # on supprime la colonne _ordre_vehicule Enfin, on réinitialise les index du DataFrame pour avoir des lignes numérotées de 0 à n-1

    return df

def prep_grav(df, colonne="grav"):
    """
    Le but est d'ordonner la gravité : indemne, blessé léger, blessé hospitalisé, tué
    Et pas indmne, tué, blessé hospitalisé, blessé léger
    """
    df[colonne] = df[colonne].replace({2: -1, 4: 2}) # on remplace temporairement 2 par -1 et 4 par 2
    df[colonne] = df[colonne].replace(-1, 4) # on remplace enfin -1 par 4

    return df

def prep_sexe(df, colonne="sexe"):
    """
    Le but est d'avoir des valeurs binaires.
    """
    df[colonne] = df[colonne].replace({1: -1, 2: 1}) # on remplace temporairement 1 par -1 et 2 par 1
    df[colonne] = df[colonne].replace(-1, 0) # on remplace enfin -1 par 0

    return df

def prep_weekend(df,
                 jour_col: str = 'jour',
                 mois_col: str = 'mois',
                 an_col:   str = 'an'):
    """
    À partir de trois colonnes entières (jour, mois, an), 
    crée une colonne datetime 'date' puis ajoute une colonne binaire 'we'
    indiquant True pour samedi et dimanche.

    :param df: DataFrame Pandas contenant les colonnes jour, mois et an.
    :param jour_col: nom de la colonne des jours (entiers 1–31).
    :param mois_col: nom de la colonne des mois (entiers 1–12).
    :param an_col:   nom de la colonne des années (AAAA).
    :return: le même DataFrame,
             avec en plus :
             - une colonne 'date' de type datetime64[ns],
             - une colonne 'we' booléenne (True si week-end).
    """
    # 1) Construire la colonne datetime
    df['date'] = pd.to_datetime(
        dict(year = df[an_col],
             month = df[mois_col],
             day = df[jour_col]),
        errors='coerce'  # NaT si date invalide
    )

    # 2) Détecter le week-end (samedi=5, dimanche=6)
    df['we'] = df['date'].dt.dayofweek >= 5
    df.drop(columns=['date'], inplace=True)

    return df

def prep_hrmn(df, column: str = 'hrmn'):
    """
    À partir d'une colonne (hrmn) au format 'HH:MM',
    convertit cette colonne en entier (0–23).

    :param df: DataFrame Pandas contenant la colonne hrmn.
    :param column: nom de la colonne des heures (format 'HH:MM').
    :return: le même DataFrame, avec la colonne hrmn remplacée par l'heure entière.
    """
    # 1) Conversion en datetime (les chaînes invalides deviennent NaT)
    times = pd.to_datetime(df[column], format='%H:%M', errors='coerce')

    # 2) Extraction de l'heure (0–23)
    df[column] = times.dt.hour

    return df

def prep_lum(df, column: str = 'lum'):
    """
    Réodronne les élments de la colonne 'lum' (lumière).
    
    """

    # 2) Réordonne les valeurs
    df[column] = df[column].replace({1: 1, 2: 2, 5: 3, 4: 4, 3: 5})

    return df

def One_hot_encoded(df,columns) :
    """
    One hot encoding for categorical variables
    :param df: DataFrame
    :param columns: list of columns to be one hot encoded
    :return: DataFrame with one hot encoded columns
    """
    
    # Check if the columns are in the DataFrame
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
    
    for column in columns:
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, dummies], axis=1)
        df.drop(column, axis=1, inplace=True)
    return df

def remove_erreurs(df):
    """
    Retire toutes les lignes d'un DataFrame pandas où au moins un champ contient '#VALEURMULTI'.
    
    Parameters:
    df (pandas.DataFrame): Le DataFrame d'entrée
    
    Returns:
    pandas.DataFrame: Le DataFrame filtré sans les lignes contenant '#VALEURMULTI'
    """
    # Vérifier si au moins une valeur contient '#VALEURMULTI' dans toutes les colonnes
    # Le ~ inverse la condition pour garder les lignes qui n'ont pas '#VALEURMULTI'
    filtered_df = df[~(df == '#VALEURMULTI').any(axis=1)]
    filtered_df = filtered_df[~(filtered_df == '#ERREUR').any(axis=1)]
    
    return filtered_df

def remove_rows_with_specific_values(df):
    """
    Retire les lignes avec des valeurs spécifiques selon les colonnes.
    
    Parameters:
    df (pandas.DataFrame): Le DataFrame d'entrée
    
    Returns:
    pandas.DataFrame: Le DataFrame filtré
    """
    df_result = df.copy()
    
    # Colonnes où il faut supprimer la valeur 9
    cols_remove_9 = ["catr", "surf", "infra", "situ", "obsm", "motor", "situ1"]
    
    # Supprimer les lignes où au moins une de ces colonnes vaut 9
    for col in cols_remove_9:
        if col in df_result.columns:
            df_result = df_result[df_result[col] != 9]
    
    # Supprimer les lignes où "manv" vaut 26
    if "manv" in df_result.columns:
        df_result = df_result[df_result["manv"] != 26]
    
    # Supprimer les lignes où "catv" vaut 0, 4, 5, 6, 8, 9, 11, 12, 18, 19, 99
    if "catv" in df_result.columns:
        catv_values_to_remove = [0, 4, 5, 6, 8, 9, 11, 12, 18, 19, 99]
        df_result = df_result[~df_result["catv"].isin(catv_values_to_remove)]
    
    return df_result


def remove_rows_with_minus_one(df):
    """
    Retire toutes les lignes d'un DataFrame pandas où au moins un champ vaut -1
    dans toutes les colonnes.
    
    Parameters:
    df (pandas.DataFrame): Le DataFrame d'entrée
    
    Returns:
    pandas.DataFrame: Le DataFrame filtré sans les lignes contenant -1
    """
    # Vérifier si au moins une valeur vaut -1 dans toutes les colonnes
    # Le ~ inverse la condition pour garder les lignes qui n'ont pas de -1
    filtered_df = df[~(df == -1).any(axis=1)]
    
    return filtered_df

def remove_unwanted_columns(df):
    """
    Supprime les colonnes spécifiées du DataFrame.
    
    Parameters:
    df (pandas.DataFrame): Le DataFrame d'entrée
    
    Returns:
    pandas.DataFrame: Le DataFrame sans les colonnes spécifiées
    """
    columns_to_remove = ["dep", "com", "adr", "voie", "v1", "v2", "pr1", "pr", 
                        "lartpc", "larrout", "id_vehicule", "id_usager", "num_veh_x", "num_veh_y", 
                        "trajet", "secu2", "secu3", "locp", "actp", "etatp", "occutc", "jour", "mois"]
    
    # Filtrer pour ne garder que les colonnes qui existent dans le DataFrame
    existing_cols_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    # Supprimer les colonnes existantes
    df_result = df.drop(columns=existing_cols_to_remove)
    
    return df_result, existing_cols_to_remove

def clean_dataframe(df):
    """
    Fonction combinée qui applique tous les nettoyages.
    
    Parameters:
    df (pandas.DataFrame): Le DataFrame d'entrée
    
    Returns:
    pandas.DataFrame: Le DataFrame nettoyé
    dict: Statistiques du nettoyage
    """
    initial_count = len(df)
    initial_columns = len(df.columns)
    
    # Suppression des colonnes non désirées
    df_clean, removed_columns = remove_unwanted_columns(df)
    after_column_removal = len(df_clean.columns)
    
    # Suppression des -1
    df_clean = remove_rows_with_minus_one(df_clean)
    after_minus_one = len(df_clean)
    
    # Suppression des valeurs spécifiques
    df_clean = remove_rows_with_specific_values(df_clean)
    final_count = len(df_clean)
    final_columns = len(df_clean.columns)
    
    df_clean = remove_erreurs(df_clean)  # Retire les lignes avec '#VALEURMULTI', '#ERREUR', etc.
    
    stats = {
        "lignes_initiales": initial_count,
        "colonnes_initiales": initial_columns,
        "colonnes_supprimees": removed_columns,
        "colonnes_apres_suppression": after_column_removal,
        "lignes_apres_suppression_-1": after_minus_one,
        "lignes_supprimees_-1": initial_count - after_minus_one,
        "lignes_finales": final_count,
        "colonnes_finales": final_columns,
        "lignes_supprimees_valeurs_specifiques": after_minus_one - final_count,
        "total_lignes_supprimees": initial_count - final_count,
        "total_colonnes_supprimees": initial_columns - final_columns,
        "pourcentage_lignes_conservees": round((final_count / initial_count) * 100, 2) if initial_count > 0 else 0,
        "pourcentage_colonnes_conservees": round((final_columns / initial_columns) * 100, 2) if initial_columns > 0 else 0
    }
    
    return df_clean, stats

def prep_lat_long(df, lat_col='lat', long_col='long'):
    """
    Applique des limites géographiques fixes pour la France métropolitaine + Corse, retire les outre-mers.
    
    Parameters:
    df (pandas.DataFrame): Le DataFrame d'entrée
    lat_col (str): Nom de la colonne latitude
    long_col (str): Nom de la colonne longitude
    
    Returns:
    pandas.DataFrame: Le DataFrame filtré pour la France métropolitaine + Corse
    """
    
    # Create a copy to avoid modifying the original
    df_result = df.copy()
    
    # Replace commas with dots for European decimal notation, then convert to numeric
    df_result[lat_col] = df_result[lat_col].astype(str).str.replace(',', '.', regex=False)
    df_result[long_col] = df_result[long_col].astype(str).str.replace(',', '.', regex=False)
    
    # Convert columns to numeric, forcing errors to NaN
    df_result[lat_col] = pd.to_numeric(df_result[lat_col], errors='coerce')
    df_result[long_col] = pd.to_numeric(df_result[long_col], errors='coerce')
    
    # Remove rows where lat or long is NaN
    df_result = df_result.dropna(subset=[lat_col, long_col])
    
    # Limites approximatives de la France métropolitaine + Corse
    lat_min, lat_max = 41.0, 51.5  # Du sud de la Corse au nord de la France
    long_min, long_max = -5.5, 10.0  # De l'ouest de la Bretagne à l'est de l'Alsace/Corse
    
    # Filtrage
    mask = (
        (df_result[lat_col] >= lat_min) & (df_result[lat_col] <= lat_max) &
        (df_result[long_col] >= long_min) & (df_result[long_col] <= long_max)
    )
    
    return df_result[mask]

def prep_An_nais(df, birth_col='an_nais', year_col='an'):
    """
    Remplace la colonne d'année de naissance par une colonne âge.
    
    Parameters:
    df (pandas.DataFrame): Le DataFrame d'entrée  
    birth_col (int): Nom de la colonne année de naissance
    year_col (int): Nom de la colonne contenant l'année de référence
    
    Returns:
    pandas.DataFrame: Le DataFrame modifié
    """
    df_result = df.copy()
    
    if birth_col not in df_result.columns:
        raise ValueError(f"La colonne '{birth_col}' n'existe pas dans le DataFrame")
    
    if year_col not in df_result.columns:
        raise ValueError(f"La colonne '{year_col}' n'existe pas dans le DataFrame")
    
    # Calcul et remplacement direct
    df_result['age'] = df_result[year_col] - df_result[birth_col]
    df_result = df_result.drop(columns=[birth_col, year_col])
    
    return df_result

# # Time to use these functions

# # %%
# # Load the data
# df = pd.read_csv("../data/accidents_2021-2023.csv")

# print(f"Initial dataset shape: {df.shape}")

# # %%
# print("="*50)
# print("APPLYING PREPROCESSING FUNCTIONS")
# print("="*50)

# # Apply all prep__ functions first
# print("\n1. Applying prep_catv...")
# df = prep_catv(df)
# print(f"   Shape after prep_catv: {df.shape}")

# # %%
# print("\n2. Applying prep_grav...")
# df = prep_grav(df)
# print(f"   Shape after prep_grav: {df.shape}")

# # %%
# print("\n3. Applying prep_sexe...")
# df = prep_sexe(df)
# print(f"   Shape after prep_sexe: {df.shape}")

# # %%
# print("\n4. Applying prep_weekend...")
# df = prep_weekend(df)
# print(f"   Shape after prep_weekend: {df.shape}")

# # %%
# print("\n5. Applying prep_hrmn...")
# df = prep_hrmn(df)
# print(f"   Shape after prep_hrmn: {df.shape}")

# # %%
# print("\n6. Applying prep_lum...")
# df = prep_lum(df)
# print(f"   Shape after prep_lum: {df.shape}")

# # %%
# print("\n7. Applying prep_lat_long...")
# df = prep_lat_long(df)
# print(f"   Shape after prep_lat_long: {df.shape}")

# # %%
# print("\n8. Applying prep_An_nais...")
# df = prep_An_nais(df)
# print(f"   Shape after prep_An_nais: {df.shape}")

# # %%
# print("="*50)
# print("CLEANING DATA")
# print("="*50)

# # Apply cleaning function
# df, cleaning_stats = clean_dataframe(df)
# print(f"\nCleaning statistics:")
# for key, value in cleaning_stats.items():
#     print(f"   {key}: {value}")

# # %%
# print("="*50)
# print("ONE HOT ENCODING")
# print("="*50)

# # Only "agg" and "int" columns need one-hot encoding
# categorical_columns = ["agg", "int"]
# print(f"Columns for one-hot encoding: {categorical_columns}")

# if all(col in df.columns for col in categorical_columns):
#     df = One_hot_encoded(df, categorical_columns)
#     print(f"Shape after one-hot encoding: {df.shape}")
# else:
#     missing_cols = [col for col in categorical_columns if col not in df.columns]
#     print(f"Warning: Missing columns {missing_cols} - skipping one-hot encoding")

# # %%
# print("="*50)
# print("NORMALIZING NUMERIC COLUMNS")
# print("="*50)

# # Normalization
# df_normalized = normalize_dataframe(df)
# print(f"Final shape after normalization: {df_normalized.shape}")

# # %%
# print("="*50)
# print("PREPROCESSING COMPLETE")
# print("="*50)

# print(f"Original shape: {pd.read_csv('../data/accidents_2021-2023.csv').shape}")
# print(f"Final shape: {df_normalized.shape}")
# print(f"Columns in final dataset: {list(df_normalized.columns)}")

# # %%
# print("="*50)
# print("FINAL DATASET PREVIEW")
# print("="*50)

# # Display first few rows and basic info
# print(df_normalized.head())
# print(f"\nDataset info:")
# print(df_normalized.info())

# # %%
# # Delete existing files if they exist
# files_to_delete = [
#     Path("../data/accidents_2021-2023_clean.csv"), 
#     Path("../data/accidents_2021-2023_clean_sample1k.csv")
# ]

# for file_path in files_to_delete:
#     if file_path.exists():
#         file_path.unlink()
#         print(f"Deleted existing file: {file_path}")

# # Save the full processed dataset
# full_path = Path("../data/accidents_2021-2023_clean.csv")
# df.to_csv(full_path, index=False)
# print(f"Full dataset saved: {len(df)} rows")

# # Create a random sample of 1000 lines
# sample_path = Path("../data/accidents_2021-2023_clean_sample1k.csv")
# df_sample = df.sample(n=1000, random_state=42)
# df_sample.to_csv(sample_path, index=False)
# print(f"Sample dataset saved: {len(df_sample)} rows")

