# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:28:00 2022

@author: angel
"""

import json # Pour gérer le fichier donneesbus.json
from math import sin, cos, acos, pi
from copy import deepcopy

with open("Fichiers/donneesBus.json") as fic_donnees_bus:
    donneesBus = json.load(fic_donnees_bus)

# Création d'une liste des noms des arrêts.
noms_arrets = list(donneesBus.keys())


def nom(ind: int) -> str:
    """
    Renvoie le nom de l'arret à l'indice ind
    :param ind: Indice de l'arrêt
    :type ind: int
    :return: nom de l'arrêt
    :rtype: str
    """
    return noms_arrets[ind]


def indice_som(nom_som: str) -> int:
    """
    Renvoie l'indice de l'arrêt à partir de son nom
    :param nom_som: Nom de l'arrêt
    :type nom_som: str
    :return: indice de l'arrêt
    :rtype: int
    """
    return noms_arrets.index(nom_som)


def lattitude(nom_som: str) -> float:
    """
    Renvoie la lattitude de l'arrêt à partir de son nom
    :param nom_som: Nom de l'arrêt
    :type nom_som: str
    :return: lattitude de l'arrêt
    :rtype: float
    """
    return donneesBus[nom_som][0]


def longitude(nom_som: str) -> float:
    """
    Renvoie la longitude de l'arrêt à partir de son nom
    :param nom_som: Nom de l'arrêt
    :type nom_som: str
    :return: longitude de l'arrêt
    :rtype: float
    """
    return donneesBus[nom_som][1]


def voisin(nom_som: str) -> list:
    """
    Renvoie la liste des arrêts voisins à partir de son nom
    :param nom_som: Nom de l'arrêt
    :type nom_som: str
    :return: liste des arrêts voisins
    :rtype: list
    """
    return donneesBus[nom_som][2]


# Création de la liste d'adjacence sous forme d'une liste.
mat_bus = [
    [1 if nom_som in voisin(nom_som1) else 0 for nom_som in noms_arrets] for nom_som1 in noms_arrets
]

# Création de la liste d'adjacence sous forme d'un dictionnaire.
dict_bus = {
    nom_arret: voisin(nom_arret) for nom_arret in noms_arrets
}


def distanceGPS(latA: float, latB: float, longA: float, longB: float) -> float:
    """
    Retourne la distance en mètres entre deux points GPS.
    :param latA: lattitude du premier point
    :param latB: lattitude du deuxième point
    :param longA: longitude du premier point
    :param longB: longitude du deuxième point
    :return:
    """
    ltA = latA / 180 * pi
    ltB = latB / 180 * pi
    loA = longA / 180 * pi
    loB = longB / 180 * pi
    # Rayon de la terre en mètres (sphère IAG-GRS80)
    RT = 6378137
    # angle en radians entre les 2 points
    S = acos(round(sin(ltA) * sin(ltB) + cos(ltA) * cos(ltB) * cos(abs(loB - loA)), 14))
    # distance entre les 2 points, comptée sur un arc de grand cercle
    return S * RT


def distance_arrets(arret1: str, arret2: str) -> float:
    """
    Renvoie la distance à vol d'oiseau entre deux arrêts.
    :param arret1: nom de l'arrêt 1
    :type arret1: str
    :param arret2: nom de l'arrêt 2
    :type arret2: str
    :return: distance entre les deux arrêts
    :rtype: float
    """
    return distanceGPS(
        lattitude(arret1),
        lattitude(arret2),
        longitude(arret1),
        longitude(arret2),
    )


def distance_arc(arret1: str, arret2: str) -> float:
    """
    Renvoie la distance à vol d'oiseau entre deux arrêts s'ils sont.
    :param arret1: nom de l'arrêt 1
    :type arret1: str
    :param arret2: nom de l'arrêt 2
    :type arret2: str
    :return: distance entre les deux arrêts
    :rtype: float
    """
    return distanceGPS(
        lattitude(arret1),
        lattitude(arret2),
        longitude(arret1),
        longitude(arret2),
    ) if arret2 in voisin(arret1) else float("inf")


# Création de la matrice des poids sous forme d'une liste.
poids_bus = [
    [distance_arc(nom_som1, nom_som2) for nom_som2 in noms_arrets] for nom_som1 in noms_arrets
]

"""
Algortihme de Bellman
Deux conditions d'arrêts :
    1- i = nombre de sommet - 1
    2- Pas de changement d'une étape à l'autre

Initialisation :
    - On initialise tous les sommets à (inf, Null) sauf le sommet de départ que l'on met à (0, Null)
"""
def bellman(arret_dep: str, arret_arriv: str) -> tuple:
    """
    Renvoie la distance la plus courte entre deux arrêts grâce à l'algorithme de Belmann.
    :param arret_dep: nom de l'arrêt de départ
    :type arret_dep: str
    :param arret_arriv: nom de l'arrêt d'arrivée
    :type arret_arriv: str
    :return: tuple(listeArrets, distance)
    :rtype: tuple
    """
    # Initialisation de la liste des distances
    distances_precedents = {sommet: [float('inf'), None] for sommet in noms_arrets}
    # Pour le sommet de départ, on met la distance à 0.
    distances_precedents[arret_dep][0] = 0

    # Fonction qui relache les sommets et renvoie un booléen.
    # Ce booléen est mit à True si on a trouvé une distance plus courte, sinon il est mit à False.
    def relachement(sommet1: str, sommet2: str) -> bool:
        """
        Relachement d'un sommet.
        Cf. https://fr.wikipedia.org/wiki/Algorithme_de_Bellman
        :param sommet1: sommet 1
        :param sommet2: sommet 2
        :return:
        """
        if distances_precedents[sommet1][0] + distance_arc(sommet1, sommet2) < distances_precedents[sommet2][0]:
            distances_precedents[sommet2][0] = distances_precedents[sommet1][0] + distance_arc(sommet1, sommet2)
            distances_precedents[sommet2][1] = sommet1
            return True
        return False

    # De base on initialise la variable contenant le booléen du changement à False.
    changement = False

    # Boucle for allant de 0 à nombre de sommets - 1
    for i in range(0, len(noms_arrets) - 2):
        # On parcourt tous les sommets
        for sommet_1 in noms_arrets:
            # On parcourt tous les voisins du sommet
            for sommet_2 in voisin(sommet_1):
                # On relache le sommet
                if changement and not relachement(sommet_1, sommet_2):
                    break
                changement = relachement(sommet_1, sommet_2)


    sommetArr = distances_precedents[arret_arriv][1]
    print(f"{sommetArr=}")
    listeArrets = [sommetArr]
    while sommetArr != arret_dep:
        sommetArr = distances_precedents[sommetArr][1]
        listeArrets.append(sommetArr)

    listeArrets = [arret_arriv] + listeArrets

    print(f"Pour aller de {arret_dep} à {arret_arriv}, il y a {distances_precedents[arret_arriv][0]} et il faut passer par les arrêts {listeArrets}.")
    

    return listeArrets, distances_precedents[arret_arriv][0]

arr2 = voisin(noms_arrets[64])[0]

print(bellman(noms_arrets[1], arr2))
