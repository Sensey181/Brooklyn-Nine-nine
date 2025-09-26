import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from pathlib import Path
import shutil

# wip

# Configuration
tf.random.set_seed(42)
np.random.seed(42)

# def focal_loss_multiclass(alpha=None, gamma=2.0): # devait aider pour les classes deséquilibrées, très décevant
#     if alpha is None:
#         alpha = [1.0, 1.0, 3.0, 15.0]  # Poids basés sur vos ratios
    
#     def focal_loss_fixed(y_true, y_pred):
#         epsilon = tf.keras.backend.epsilon()
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
#         y_true = tf.cast(y_true, tf.int32)
#         y_true_onehot = tf.one_hot(y_true, depth=4)
        
#         # Calcul focal loss
#         ce = -y_true_onehot * tf.math.log(y_pred)
#         weight = tf.pow(1 - y_pred, gamma)
        
#         # Application des poids alpha
#         alpha_tensor = tf.constant(alpha, dtype=tf.float32)
#         alpha_weight = tf.gather(alpha_tensor, y_true)
#         alpha_weight = tf.expand_dims(alpha_weight, axis=-1)
        
#         focal = alpha_weight * weight * ce
#         return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))
    
#     return focal_loss_fixed

def load_accident_data(file_path):
   """Charge et pretraite les donnees d'accidents"""
   print("Chargement des donnees...")
   data = pd.read_csv(file_path)
   
   print(f"Dataset: {data.shape[0]} echantillons, {data.shape[1]} variables")
   print(f"Distribution gravite: {data['grav'].value_counts().sort_index().to_dict()}")
   
   # Conversion variables booleennes
   bool_cols = ['we', 'agg_1', 'agg_2', 'int_1', 'int_2', 'int_3', 
               'int_4', 'int_5', 'int_6', 'int_7', 'int_8', 'int_9']
   
   for col in bool_cols:
       if col in data.columns:
           data[col] = data[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    
   # Sous-échantillonnage équilibré
   distr = data['grav'].value_counts().sort_index()
   target_per_class = distr.min() * 5  # on maximise le sampling à 5x la classe rare (4, décès)
                                       # (pour toute la donnee, vaut autour de 8k x 5 = 40k)
   
   balanced_data = []
   for grav in [1, 2, 3, 4]:
       subset = data[data['grav'] == grav]
       if grav in [1, 2]:  # Classes majoritaires
           sampled = subset.sample(n=target_per_class, random_state=42)
       elif grav == 3:     # Classe moyenne
           sampled = subset.sample(n=min(len(subset), target_per_class), random_state=42)
       else:               # Classe minoritaire (4)
           sampled = subset  # Garder tout
       
       balanced_data.append(sampled)
       print(f"Gravité {grav}: {len(subset)} -> {len(sampled)}")
   
   final_data = pd.concat(balanced_data, ignore_index=True)
   
   return final_data

def prepare_features(data):
   """Prepare les features pour l'entrainement"""
   feature_cols = [
       'place', 'catu', 'sexe', 'secu1', 'senc', 'catv', 'obs', 'obsm',
       'choc', 'manv', 'motor', 'catr', 'circ', 'nbv', 'vosp', 'plan',
       'surf', 'infra', 'situ', 'vma', 'hrmn', 'lum', 'atm', 'col',
       'we', 'age', 'agg_1', 'agg_2', 'int_1', 'int_2', 'int_3',
       'int_4', 'int_5', 'int_6', 'int_7', 'int_8', 'int_9'
   ]
   
   available_features = [col for col in feature_cols if col in data.columns]
   print(f"Features utilisees: {len(available_features)} variables")
   
   X = data[available_features].fillna(0)
   y = data['grav'] - 1  # Base 0
   
   return X, y

def split_and_scale_data(X, y):
   """Division train/test et normalisation"""
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
   
   return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def build_model(hp):
   """Construction du modele avec hyperparametres"""
   model = keras.Sequential()
   
   # Couche entree
   model.add(layers.Dense(
       units=hp.Int('units_1', min_value=128, max_value=1024, step=32),
       activation=hp.Choice('activation', ['relu']),
       input_shape=(X_train_scaled.shape[1],)
   ))
   
   model.add(layers.Dropout(
       rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)
   ))
   
   # Couches cachees
   for i in range(hp.Int('num_layers', min_value=2, max_value=5)):
       model.add(layers.Dense(
           units=hp.Int(f'units_{i+2}', min_value=64, max_value=1024, step=32),
           activation=hp.get('activation')
       ))
       
       model.add(layers.Dropout(
           rate=hp.Float(f'dropout_{i+2}', min_value=0.0, max_value=0.5, step=0.1)
       ))
   
   # Couche sortie
   model.add(layers.Dense(4, activation='softmax'))
   
   # Compilation
   learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
   optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
   
   if optimizer == 'adam':
       opt = keras.optimizers.Adam(learning_rate=learning_rate)
   else:
       opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
   
   model.compile(
       optimizer=opt,
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy']
   )
   
   return model

def run_tuning(X_train_scaled, y_train, max_trials=15, epochs=30):
   """Lance l'optimisation des hyperparametres"""
   
   print("Demarrage optimisation hyperparametres...")
   tuner = kt.BayesianOptimization(
       build_model,
       objective='val_accuracy',
       max_trials=max_trials,
       directory='tuning_results',
       project_name='accident_severity'
   )
   
   callbacks = [
       keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
       keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
   ]
   
   tuner.search(
       X_train_scaled, y_train,
       epochs=epochs,
       validation_split=0.2,
       callbacks=callbacks,
       verbose=1
   )
   
   return tuner

def get_best_hyperparameters(tuner):
   """Recupere et affiche les meilleurs hyperparametres"""
   best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
   
   print("\nMEILLEURS HYPERPARAMETRES:")
   print("-" * 40)
   print(f"Units couche 1: {best_hps.get('units_1')}")
   print(f"Activation: {best_hps.get('activation')}")
   print(f"Optimiseur: {best_hps.get('optimizer')}")
   print(f"Learning rate: {best_hps.get('learning_rate'):.6f}")
   print(f"Nombre couches: {best_hps.get('num_layers') + 1}")
   print(f"Dropout 1: {best_hps.get('dropout_1')}")
   
   for i in range(best_hps.get('num_layers')):
       print(f"Units couche {i+2}: {best_hps.get(f'units_{i+2}')}")
       print(f"Dropout {i+2}: {best_hps.get(f'dropout_{i+2}')}")
   
   return best_hps

def train_best_model(tuner, best_hps, X_train_scaled, y_train, epochs=50):
   """Entraine le meilleur modele"""
   print("\nEntrainement du meilleur modele...")
   
   model = tuner.hypermodel.build(best_hps)
   
   callbacks = [
       keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
       keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7)
   ]
   
   history = model.fit(
       X_train_scaled, y_train,
       epochs=epochs,
       validation_split=0.2,
       callbacks=callbacks,
       verbose=1
   )
   
   return model, history

def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):
   """Evaluation du modele"""
   print("\nEVALUATION DU MODELE:")
   print("-" * 30)
   
   train_loss, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
   test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
   
   print(f"Precision train: {train_acc:.4f}")
   print(f"Precision test: {test_acc:.4f}")
   print(f"Perte train: {train_loss:.4f}")
   print(f"Perte test: {test_loss:.4f}")
   
   y_pred = model.predict(X_test_scaled)
   y_pred_classes = np.argmax(y_pred, axis=1)
   
   print("\nRAPPORT CLASSIFICATION:")
   target_names = ['Gravite 1', 'Gravite 2', 'Gravite 3', 'Gravite 4']
   print(classification_report(y_test, y_pred_classes, target_names=target_names))
   
   print("\nMATRICE CONFUSION:")
   cm = confusion_matrix(y_test, y_pred_classes)
   print(cm)
   
   return {
       'train_acc': train_acc,
       'test_acc': test_acc,
       'predictions': y_pred,
       'confusion_matrix': cm
   }

# Programme principal
if __name__ == "__main__":
   
   print("ANALYSE ACCIDENTS AVEC KERAS TUNER")
   print("=" * 50)
   
   # Chargement et preparation donnees
   data_path = '../data/accidents_2021-2023_clean.csv'
   data = load_accident_data(data_path)
   X, y = prepare_features(data)
   X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
   
   # Optimisation hyperparametres
   tuner = run_tuning(X_train_scaled, y_train, max_trials=32, epochs=40)
   best_hps = get_best_hyperparameters(tuner)
   
   # Entrainement modele final
   best_model, history = train_best_model(tuner, best_hps, X_train_scaled, y_train, epochs=200)
   
   # Evaluation
   results = evaluate_model(best_model, X_train_scaled, X_test_scaled, y_train, y_test)
   
   # Sauvegarde
   best_model.save('accident_model.h5')
   print("\nModele sauvegarde: accident_model.h5")
   
   print("\nAnalyse terminee.")