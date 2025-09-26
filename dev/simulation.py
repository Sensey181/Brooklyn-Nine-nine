#%%
import pandas as pd
import requests
from data_prep import prep_weekend, prep_An_nais, prep_lat_long, prep_hrmn, prep_lum, prep_catv, normalize_dataframe
from joblib import load
import tensorflow as tf
import numpy as np

simulation = {}

#%%
## USAGERS
place = int(input(f"\nPlace occupée par l'usager dans le véhicule au moment de l'accident (voir documentation simulation) : "))
if place not in range(1,10):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["place"]=[place]

catu = int(input(f"\nQuelle est la catégorie de l'usager ? 1-Conducteur, 2-Passager, 3-Piéton : "))
if catu not in range(1,4):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["catu"]=[catu]
2
sexe = int(input(f"\nQuel est le sexe de l'usager ? 0-Masculin, 1-Féminin : "))
if sexe not in range(0,2):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["sexe"]=[sexe]

secu1 = int(input(f"\nPrésence et utilisation d'un équipement de sécurité ? 0-Aucun, 1-Ceinture, 2-Casque, 3-Dispositifs enfants, 4-Gilet réfléchissant, 5-Airbag, 6-Gants, 7-Gants+Airbag, 8-Non déterminable : "))
if secu1 not in range(0,9):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["secu1"]=[secu1]

## VEHICULES
senc = int(input("\nQuel est le sens de circulation ? 0-Inconnu, 1-PK ou PR ou numéro d'adresse postale croissant, 2-PK ou PR ou numéro d'adresse postale décroissant, 3-Absence de repère : "))
if senc not in range(0,4):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["senc"] = [senc]

catv = int(input("\nQuelle est la catégorie du véhicule (voir documentation) ? : "))
if catv not in [60, 50, 80, 1, 2, 30, 31, 32, 41, 42, 3, 35, 36, 33, 34, 43, 7, 10, 13, 14, 15, 16, 17, 21, 20, 37, 38, 39, 40]:
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["catv"] = [catv]

obs = int(input("\nLe véhicule a-t-il heurté un obstacle fixe (voir documentation) ? : "))
if obs not in range(0,18):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["obs"] = [obs]

obsm = int(input("\nLe véhicule a-t-il heurté un obstacle mobile ? 0-Aucun, 1-Piéton, 2-Véhicule, 4-Véhicule sur rail, 5-Animal domestique, 6-Animal sauvage : "))
if obsm not in range(0,7):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["obsm"] = [obsm]

choc = int(input("\nY a-t-il eu un point de choc initial ? 0-Aucun, 1-Avant, 2-Avant droit, 3-Avant gauche, 4-Arrière, 5-Arrière droit, 6-Arrière gauche, 7-Côté droit, 8-Côté gauche, 9-Chocs multiples (tonneaux) : "))
if choc not in range(0,10):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["choc"] = [choc]

manv = int(input("\nQuelle est la manoeuvre effectuée avant l'accident (voir documentation) ? : "))
if manv not in range(0,27):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["manv"] = [manv]

motor = int(input("\nQuel est le type de motorisation du véhicule ? 0-Inconnue, 1-Hydrocarbures, 2-Hybride électrique, 3-Electrique, 4-Hydrogène, 5-Humaine : "))
if motor not in range(0,6):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["motor"] = [motor]

## LIEUX
catr = int(input(f"\nQuelle est la catégorie de route ? 1-Autoroute, 2-Route nationale, 3-Route départementale, 4-Voie Communale, 5-Hors réseau public, 6-Parc de stationnement ouvert à la circulation publique, 7-Routes de métropole urbaine : "))
if catr not in range(1,8):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["catr"]=[catr]

circ = int(input(f"\nQuel est le régime de circulation ? 1-Sens unique, 2-Bidirectionnelle, 3-Chaussées séparées, 4-Voies d'affectation variable : "))
if circ not in range(1,5):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["circ"]=[circ]

nbv = int(input(f"\nQuel est le nombre de voies de circulation ? : "))
simulation["nbv"]=[nbv]

vosp = int(input("\nExiste-il une voie réservée ? 0-Sans objet, 1-Piste cyclable, 2-Bande cyclable, 3-Voie réservée : "))
if vosp not in range(0,4):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["vosp"] = [vosp]

prof = int(input("\nQuelle est la forme de la route ? 1-Plat, 2-Pente, 3-Sommet de côte, 4-Bas de côté : "))
if prof not in range(1,5):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["prof"] = [prof]

plan = int(input("\nQuel est le tracé de la route ? 1-Rectiligne, 2-Courbe à gauche, 3-Courbe à droite, 4-En S : "))
if plan not in range(1,5):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["plan"] = [plan]

surf = int(input("\nQuel est l'état de la surface de la route ? 1-Normale, 2-Mouillée, 3-Flaques, 4-Innondée, 5-Enneigée, 6-Boue, 7-Verglacée, 8-Huileuse : "))
if surf not in range(1,9):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["surf"] = [surf]

infra = int(input("\nY a-t-il un aménagement ou une infrastructure ? 0-Aucun, 1-Souterrain/Tunnel, 2-Pont, 3-Bretelle/Raccordement, 4-Voie ferrée, 5-Carrefour aménagé, 6-Zone piétonne, 7-Zone de péage, 8-Chantier : "))
if infra not in range(0,9):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["infra"] = [infra]

situ = int(input("\nOù se situe l'accident ? 0-Aucun, 1-Chaussée, 2-Bande d'arrêt d'urgence, 3-Accotement, 4-Trottoir, 5-Piste cyclable, 6-Autre voie spéciale : "))
if situ not in range(0,7):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["situ"] = [situ]

vma = int(input("\nQuelle est la vitesse maximale autorisée sur le lieu et au moment de l'accident ? : "))
simulation["vma"] = [vma]

## CARACTERISTIQUES
jour = int(input("\nQuel jour sommes-nous (le numéro) ? : "))
if jour not in range(1,32):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["jour"] = [jour]

mois = int(input("\nQuel mois sommes-nous ? : "))
if mois not in range(1,13):
    raise Exception("Données d'entrée invalide, relis la question !")
simulation["mois"] = [mois]

an = int(input("\nQuelle est l'année 'XXXX' ? : "))
simulation["an"] = [an]

# Conversion du dict en DataFrame pandas ici
simulation = pd.DataFrame(simulation)

hrmn = str(input("\nQuelle heure est-il 'HH:MM' ? : "))
simulation["hrmn"] = [hrmn]

lum = int(input("\nQuelles sont les conditions d'éclairage dans lesquelles l'accident s'est produit ? 1-Plein jour, 2-Crépuscule/aube, 3-Nuit sans éclairage public, 4-Nuit avec éclairage public non allumé, 5-Nuit avec éclairage public allumé : "))
simulation["lum"] = [lum]

atm = int(input("\nQuelles sont les conditions atmosphériques ? 1-Normale, 2-Pluie légère, 3-Pluie forte, 4-Neige/Grêle, 5-Brouillard/Fumée, 6-Vent fort/Tempête, 7-Temps éblouissant, 8-Temps couvert : "))
simulation["atm"] = [atm]

col = int(input("\nQuel est le type de collision ? 1-Deux véhicules frontalement, 2-Deux véhicules par l'arrière, 3-Deux véhicules par le côté, 4-Trois véhicules et plus en chaîne, 5-Trois véhicules et plus collisions multiples, 6-Autre collision, 7-Sans collision : "))
simulation["col"] = [col]

adresse=str(input("\nQuelle est l'adresse ?"))

def adresse_vers_coords(adresse):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': adresse,
        'format': 'json',
        'limit': 1
    }
    headers = {
        'User-Agent': 'MonProgrammePython/1.0 (email@exemple.com)'
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            latitude = float(data[0]['lat'])
            longitude = float(data[0]['lon'])
            return latitude, longitude
        else:
            print("Adresse introuvable.")
            return None, None
    else:
        print("Erreur API :", response.status_code)
        return None, None
       
latitude, longitude = adresse_vers_coords(adresse)

# Ajouter lat et long à la DataFrame
simulation["lat"] = latitude
simulation["long"] = longitude

simulation = prep_weekend(simulation, jour_col="jour", mois_col="mois", an_col="an") 

an_nais = int(input(f"\nQuelle est l'année de naissance de l'usager ? : "))
simulation["an_nais"]=[an_nais]
simulation = prep_An_nais(simulation, 'an_nais', 'an')

agg = int(input("\nHors agglomération 0 ou en agglomération 1 ? : "))
if agg == 0 :
    simulation["agg_1"] = [True]
    simulation["agg_2"] = [False]
else:
    simulation["agg_1"] = [False]
    simulation["agg_2"] = [True]

int_ = int(input("\nQuel est le type d'intersection ? 1-Hors intersection, 2-Intersection en X, 3-Intersection en T, 4-Intersection en T, 5-Intersection à plus de 4 branches, 6-Giratoire, 7-Place, 8-Passage à niveau : "))
for i in range(1, 10):
    simulation[f"int_{i}"] = [i == int_]

## Préparation des données

simulation = prep_catv(simulation,"catv")
simulation = prep_lat_long(simulation, 'lat', 'long')
simulation = prep_hrmn(simulation,'hrmn')
simulation = prep_lum(simulation, 'lum')

# Suppression colonnes temporaires inutiles
simulation.drop(columns=["jour", "mois"], inplace=True)

pd.set_option('display.max_rows', None)     # Affiche toutes les lignes
pd.set_option('display.max_columns', None)  # Affiche toutes les colonnes
pd.set_option('display.width', None)        # Ne tronque pas selon la largeur de l'écran
pd.set_option('display.max_colwidth', None) # Ne tronque pas le contenu des cellules
print(simulation)

#simulation = normalize_dataframe(simulation)

#print(simulation)

#%%

print(simulation.columns)

## Récupérer les modèles
model = tf.keras.models.load_model("dev/accident_model.h5")

## Les tester sur notre simu

prediction = model.predict(simulation)  # ou data
print("Prédiction :", prediction)

# %%
