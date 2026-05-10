# -*- coding: utf-8 -*-
"""
Functions definition
"""

# Imports 
import pandas as pd 
import numpy as np
import os
import sys


#-----------------------------------------------------------------------
# Functions 


# Data loading
def path_check (path): 
    if os.path.exists(path):
        return True
    else : 
        return False
    
    
def read_datas(full_path):
    " Chargement du csv avec correction des bugs de format prenant en compte l'ecodage, la séparation des valeurs par ';' et la conversion de strings en flottants"    
    if path_check(full_path):
        raw_datas = pd.read_csv(full_path, encoding="cp1252", sep=";", decimal=',')
        return raw_datas
    else : 
        print('Fichier non trouvé - Chemin : ' + full_path + " incorrecte")
        sys.exit()
        
# Visualisation
def trace_regression(beta_0, beta_1, x):
    return beta_0 + x*beta_1


def smooth(x,y, box_percent=0.05,res=50,median=True):

    surface = max(x)-min(x)

    my_pas = np.arange(min(x),max(x),surface/res)

    box = surface*box_percent

    demi_box = box/2

    y_sortie = np.array([])

    x_sortie = np.array([])

    for myx in my_pas :

        temp = [y[i] for i in range(len(x)) if ((x[i]>=(myx-demi_box))and(x[i]<=(myx+demi_box)))]

        if median==True :

            temp_y = np.median(temp)

        else :

            temp_y = np.mean(temp)

        #print(temp_y)

        y_sortie = np.append(y_sortie,temp_y)

        #print(y_sortie)

        x_sortie = np.append(x_sortie,myx)

    return x_sortie, y_sortie
