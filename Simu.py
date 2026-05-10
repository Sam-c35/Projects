# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 14:52:58 2026

@author: Sam

Essaye de faire tourner des simulations ainsi que des moyens de la visualiser
Pour commencer, je veux prendre un cas d'étude cool donc une planète qui orbite autour d'une étoile

"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Paramètres 

M = 30 # Masse de l'étoile
m = 0.3 # Masse de la planète
x_init = 1 # Position de la planete
y_init = 0 # Position de la planete
Vx_init = 0 # Vitesse en X de la planete
Vy_init = 2.8 # Vitesse en Y de la planete
constante_gravitationnelle = 1 #6.6743*10**(-11) #Constante gravitationnelle en N.m²/kg²

delta_t = 0.01 #Définition du pas de temps
duree_simu = 10 #Durée maximum de la simulation


#%% Functions 

def force_grav(M_et, m_plan, rayon, position, gravitation): 
    return -gravitation*M_et*m_plan*position/rayon**3

def acceleration(M_et, position, rayon, gravitation):
    ax = -gravitation*M_et*position[0]/rayon**3
    ay = -gravitation*M_et*position[1]/rayon**3
    return np.array([ax, ay])

def distance(r): 
    return np.linalg.norm(r)



#%% Main

r = np.array([x_init,y_init])  # Positions initiales
v = np.array([Vx_init,Vy_init]) # Vitesses initiales

positions= [] # Liste de toutes les positions calculées par pas de temps 
vitesses = [] # Liste de toutes les vitesses calculées à chaque pas de temps
temps = [i*delta_t for i in range(int(duree_simu/delta_t)+1)]

# Initialisastion de nos valeurs initiales dans nos listes de vitesses et positions
positions.append(r)
vitesses.append(v)



# Si on choisi d'utiliser la méthode d'Euler
for j in range(int(duree_simu/delta_t)) : 
    "on utilise deux variables qui parcourent d'une part tous nos pas de temps (t) et d'autre part l'index correspondant à chaque pas de temps (j)"
    
    vitesses.append(vitesses[j]+acceleration(M, positions[j], distance(r), constante_gravitationnelle)*delta_t)
    positions.append(positions[j] + vitesses[j+1]*delta_t)
    
# Maintenant que les données sont calculées, on modifie seulement le format pour obtenir l'affichage
positions = np.array(positions)

x = positions[:,0]
y = positions[:,1]

plt.figure()
plt.plot(x, y)
plt.scatter(x[0], y[0], color='green', label='Position initiale')
plt.scatter(0, 0, color='orange', label='Étoile')
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.show()

# %% Main 2 

" Dans ce main, on veut cette fois une animation dynamique à l'aide de matplotlib.animation en codant la méthode d'euler"

r = np.array([x_init,y_init])  # Positions initiales
v = np.array([Vx_init,Vy_init]) # Vitesses initiales

positions = []

def update(frame): 
    global r, v
    
    a = acceleration(M, r, distance(r), constante_gravitationnelle)
    
    v = v + a*delta_t
    r = r + v*delta_t
    
    positions.append(r.copy())
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    line.set_data(xs, ys)
    planet.set_data([r[0]], [r[1]])
    return line, planet
    
    
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

star, = ax.plot(0, 0, 'yo', markersize=10)
planet, = ax.plot([], [], 'bo')
line, = ax.plot([], [], 'b-', linewidth=1)

ani = FuncAnimation(fig, update, frames=3000, interval=10)
plt.show()

#%% Main 3 

" Dans cette section, on recommence l'étude en essayant d'utiliser une seconde méthode (méthode de Verlet) et de comparer les deux méthodes"

" On commence par coder la méthode d'euler de manière plus simple que précédement"

def euler(r, v, G, Met, delta_t):
    a = acceleration(Met, r, distance(r), G)
    
    v_new = v + a*delta_t
    r_new = r + v_new*delta_t
    
    return r_new, v_new

" On essaye ensuite de coder la méthode de verlet"

def verlet(r, v, G, Met, delta_t):
    a = acceleration(Met, r, distance(r), G)
    
    v_new = v + a*delta_t
    r_old = r - v*delta_t
    r_new = 2*r - r_old + a*delta_t**2
    
    return r_new, v_new

pos_euler = []
pos_verlet = []
temps = []

" Construction de nos vecteurs initiaux (positions et viteses, ce sont les mêmes valeurs initiales pour chaque méthode mais pour la suite de la comparaison, on doit le faire séparément"
r_euler = np.array([x_init, y_init])
r_verlet = np.array([x_init, y_init])
v_euler = np.array([Vx_init, Vy_init])
v_verlet = np.array([Vx_init, Vy_init])

t=0
while(t<2000):
    "Calcul des trajectoires/vitesses pour les deux méthodes"
    r_euler, v_euler = euler(r_euler, v_euler, constante_gravitationnelle, M, delta_t)
    r_verlet, v_verlet = verlet(r_verlet, v_verlet, constante_gravitationnelle, M, delta_t)
    
    pos_euler.append(r_euler)
    pos_verlet.append(r_verlet)
    
    temps.append(t)
    
    t+=1
    
pos_euler = np.array(pos_euler)
pos_verlet = np.array(pos_verlet)

x_euler = pos_euler[:,0]
y_euler = pos_euler[:,1]
x_verlet = pos_verlet[:,0]
y_verlet = pos_verlet[:,1]

plt.figure()
plt.plot(x_euler, y_euler, label="Euler", alpha=0.6, linewidth=0.6)
plt.plot(x_verlet, y_verlet, label="Verlet", linewidth=0.6)
plt.legend()
plt.axis("equal")


plt.figure()
plt.plot(temps , x_verlet, label="Verlet", linewidth=0.6)
plt.plot(temps, x_euler, label="euler", linewidth=0.6)
plt.legend()
