# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:28:00 2022

@author: angel
"""

import json

from math import sin, cos, acos, pi

#A-Importer les données dans un dictionnaire donneesBus
with open("donneesBus.json") as ficDonneesBus:
    donneesBus = json.load(ficDonneesBus)
  
#B-Creer une liste nom_arrets conteant tous les noms d'arrets
nom_arrets = list(donneesBus.keys())

#C-Créer des fonctions:

#Fonction qui renvoie le nom de l'arret à partir d'un indice
def nom(ind):
    return nom_arrets[ind]

#Fonction qui renvoi l'indice de l'arret à partir d'un nom d'arret
def indice_som(nom_som):
    return nom_arrets.index(nom_som)

#Fonction qui renvoie la lattitude d'un arret
def lattitude(nom_som):
    return donneesBus[nom_som][0]

#Fonction qui renvoie la longitude d'un arret
def longitude(nom_som):
    return donneesBus[nom_som][1]

#Fonction qui renvoie les voisins d'un arret 
def voisin(nom_som):
        return donneesBus[nom_som][2]
    
#D-

#Liste d'adjacence par un dictionnaire dic_bus
dic_bus = {}
for i in nom_arrets:
    dic_bus[i] = voisin(i)

#Matrice d'adjacence
mat_bus = [
    [1 if i in voisin(arret) else 0 for i in nom_arrets] for arret in nom_arrets
]

#E-
 
def distanceGPS(latA,latB,longA,longB):
 # Conversions des latitudes en radians
    ltA=latA/180*pi
    ltB=latB/180*pi
    loA=longA/180*pi
    loB=longB/180*pi
    # Rayon de la terre en mètres (sphère IAG-GRS80)
    RT = 6378137
    # angle en radians entre les 2 points
    S = acos(round(sin(ltA)*sin(ltB) + cos(ltA)*cos(ltB)*cos(abs(loB-loA)),14))
    # distance entre les 2 points, comptée sur un arc de grand cercle
    return round(S*RT)

#Fonction qui renvoie la distance à vol d'oiseau entre deux arrets
def distArrets(arret1,arret2):
    return distanceGPS(lattitude(arret1), lattitude(arret2), longitude(arret1), longitude(arret2))

#Fonction qui renvoie la distance à vol d'oiseau entre deux arrets s'ils sont voisins, sinon renvoie 'inf' 
def distArc(arret1, arret2):
    if arret1 in voisin(arret2):
        return distanceGPS(lattitude(arret1), lattitude(arret2), longitude(arret1), longitude(arret2))
    else:
        return float('inf')
    
#F-Matrice des poids
poids_bus = [
    [distArc(arret1, arret2) for arret2 in nom_arrets] for arret1 in nom_arrets
    ]
