# -*- coding: utf-8 -*-

#Imports 

import pandas as pd 
import os 
import Functions as fct
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

" Etude d'un jeu de données du secteur residentiel"
" Une mesure effectuée toutes les 10min pour obtenir une consommation annuelle"
" 100 logements considérés comme référence"
" Consommation exprimée en kWh/an"


# variables 

data_path = "C:\\Users\Sam\Desktop\Python\Data"
data_name = "Fichiers_de_donnees_conso_annuelles_V1.csv"

#%% Parameters
" Choisi les parametres que vous souhaitez pour votre analyse"
" Permet aussi de choisir les résultats que vous souhaitez afficher"





#%%
# Main 

full_path= os.path.join(data_path, data_name)

datas = fct.read_datas(full_path)

# Renomme les colonnes pour plus de praticité
datas.columns = ["Appareil", "ID", "Conso_1", "Conso_2", "Type"]



#%% Analyse 1 
" Etude de la variation de consommation maison par maison d'une année à l'autre"

# Visualisation de la conso totale de toutes les maisons 

# Sélection des colonnes qui nous intéressent à ce stade : Numéro de logement et consomation sur les deux années de référence
df1 = datas[["ID", "Conso_1", "Conso_2"]]

conso_maisons = df1.groupby("ID")[["Conso_1", "Conso_2"]].sum()

# Tri de l'affichage du plus gros consommateur au moins gros (sur la première année)
conso_maisons = conso_maisons.sort_values(by=["Conso_1"])

# Remplacement des numéros de logement par des numéros arbitraires 
conso_maisons = conso_maisons.reset_index()

# On enlève toutes les maisons qui n'ont pas consommé sur une des deux années pour ne pas polluer les données
conso_maisons = conso_maisons[(conso_maisons["Conso_1"]>0) & (conso_maisons["Conso_2"]>0)]

# Affichage de la comparaison entre les consommations
plt.figure()
plt.scatter(conso_maisons.index, conso_maisons['Conso_1'], color='green', marker='o', label='AN1')
plt.scatter(conso_maisons.index, conso_maisons['Conso_2'], color='red', marker='o', label='AN2')
plt.xlabel("Maison")
plt.ylabel("Conso")
plt.title("Comparatif Consommation AN1/AN2 par maison")
plt.legend(['Année 1', 'Année 2'])
plt.show()

# Calcul de l'écart de consommation entre l'année 1 et la 2 
conso_maisons['Ecart_annuel'] = (conso_maisons["Conso_1"] - conso_maisons["Conso_2"])


# Tentative d'étude statistique avec approximation de regression linéaire
"calcul de nos paramètres : moyennes, coefficients a et b"
x = conso_maisons["Conso_1"].values
y = conso_maisons["Ecart_annuel"].values
n=len(x)

x_mean = np.mean(x)
y_mean = np.mean(y)

" calcul de a et de b"
a = sum([(x[i]-x_mean)*(y[i]-y_mean) for i in range(n)])/sum([(x[i]-x_mean)**2 for i in range(n)])
b = y_mean - a*x_mean

x_reg = np.linspace(x.min(), x.max(), 200)
y_reg = a * x_reg + b

" Après avoir tracé la regression, on veut vérifier la qualité"
" Dans notre cas, R² = 0.004 donc on peut dire que la correlation entre les données est ridicule"

y_pred = a*x+b

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)

r2_lin = 1 - ss_res / ss_tot


plt.figure()
plt.scatter(conso_maisons["Conso_1"], conso_maisons['Ecart_annuel'], color='green', marker='o', label='Ecart annuel par maison')
plt.xlabel("Consommation lors de la première année [kWh]")
plt.ylabel("Ecart [kWh]")
plt.title("Ecart de consommation entre l'année 1 et la 2 en fonction de la consommation initiale (en valeur absolue)")
plt.plot(x_reg, y_reg, color="red", label="Régression linéaire")
plt.legend()
plt.show()



# Seconde tentative d'étude statistique en utilisant cette fois-ci une regression quadratique

Sx = np.sum(x)
Sx2 = np.sum(x**2)
Sx3 = np.sum(x**3)
Sx4 = np.sum(x**4)

Sy = np.sum(y)
Sxy = np.sum(x*y)
Sx2y = np.sum((x**2)*y)

A = np.array([
    [Sx4, Sx3, Sx2],
    [Sx3, Sx2, Sx],
    [Sx2, Sx,  n]
])

B = np.array([Sx2y, Sxy, Sy])

a, b, c = np.linalg.solve(A, B)

x_reg_2 = np.linspace(x.min(), x.max(), 200)
y_reg_2 = a*x_reg_2**2 + b*x_reg_2 + c


# Analyse des résidus 
y_pred_quad = a*x**2 + b*x + c
residus = y - y_pred_quad
plt.figure()
plt.scatter(x, residus, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Conso_1")
plt.ylabel("Résidu")
plt.title("Analyse des résidus (modèle quadratique)")
plt.show()

ss_res = np.sum((y - y_pred_quad) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)

r2_quad = 1 - ss_res / ss_tot



# Troisième étude statistique en essayant avec une regression logarithmique

mask = x > 0  # Condition du logarithme
z_log = np.log(x[mask]) # on pose z =ln(x) pour avoir à résoudre une regression linéaire entre z et y
y_log = y[mask]

z_mean = np.mean(z_log) 
y_mean_log = np.mean(y_log)

a_log = np.sum((z_log - z_mean)*(y_log - y_mean)) / np.sum((z_log - z_mean)**2)
b_log = y_mean - a_log*z_mean

x_reg_log = np.linspace(x[mask].min(), x[mask].max(), 200)
y_reg_log = a_log * np.log(x_reg_log) + b_log

plt.figure()
plt.scatter(x, y, alpha=0.6, label="Données")
plt.plot(x_reg_2, y_reg_2, color="red", label="Régression quadratique")
plt.plot(x_reg, y_reg, color="blue", label="Régression linéaire")
plt.plot(x_reg_log, y_reg_log, color="green", label="Régression logarithmique")
plt.legend()
plt.show()

y_pred_log = a_log*np.log(x) + b_log
ss_res_log = np.sum((y - y_pred_log) ** 2)
ss_tot_log = np.sum((y - y_mean_log) ** 2)

r2_log = 1 - ss_res_log / ss_tot_log




#%% Etude des appareils ce consommation 

# On ne garde que les maisons qui ont participé aux deux années de l'étude

# on somme la consommation de chaque appareil
conso_appareil = datas.groupby("Appareil")[["Conso_1", "Conso_2"]].sum()

# On enlève toutes les maisons qui n'ont pas consommé sur une des deux années pour ne pas polluer les données
" Le but est de ne garder que les maisons qui ont participé aux deux années de l'étude"
conso_maisons = conso_maisons[(conso_maisons["Conso_1"]>0) & (conso_maisons["Conso_2"]>0)]

# Pour un meilleur affichage, on enlève la consommation général de la maison (on ne regarde que les appareils au détail)
conso_appareil = conso_appareil.drop(labels="Général", axis=0)

# Tri du plus au moins consommateur
conso_appareil = conso_appareil.sort_values(by=["Conso_1"], ascending=False)

# Même affichage que précédement mais en comparant avec l'ajout d'une deuxième série de barres la conso de l'année 2

# Bar width and x locations
w, x = 0.4, np.arange(len(conso_appareil.index))

fig, ax = plt.subplots()
ax.bar(x - w/2, conso_appareil['Conso_1'], width=w, label='Année 1')
ax.bar(x + w/2, conso_appareil['Conso_2'], width=w, label='Année 2')

ax.set_xticks(x)
ax.set_xticklabels(conso_appareil.index)
ax.set_ylabel('Consommation [kWh]')
ax.set_title('Comparaison consommation entre les deux années')
ax.legend()
plt.xticks(rotation=45, fontsize=9, ha="right", rotation_mode="anchor")
plt.xlim(-1, len(conso_appareil.index))
plt.show()





