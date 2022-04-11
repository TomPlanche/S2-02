# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:57:26 2022

@author: angel
"""

"""
Conditions d'arrêt Bellman :
    1- k = nbSommet - 1
    2- Pas de changement d'une étape a l'autre
    
Initialisation:
    Sommet(Distance,Predécesseur)
    -tout a (inf, Null)
    -sauf 1er : (0, Null)

A = matrice d'adjacence

Relachement (element[i] de A ex (1,3)):
    dist(1) + poids(1,3) = qqchose
    dist(3) = autre chose
    Si qqchose < autre chose alors changement
"""

    