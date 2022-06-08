# -*- coding: utf-8 -*-
from ressources import graphics as gr

global LARGEUR_FENETRE
global HAUTEUR_FENETRE

LARGEUR_FENETRE = 900
HAUTEUR_FENETRE = 900


import json  # Pour gérer le fichier donneesbus.json
from math import sin, cos, acos, pi, sqrt
from time import sleep

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


def extract_min(liste):
    """
    Retourne le sommet de poids minimum de la liste liste.
    Cf. Cours Mme Bruyère
    
    :param liste : liste des arrets
    :type liste: list
    :return: le sommet de poids minimum
    :rtype: int
    """
    minS = float("inf")
    valS = float("inf")

    for i in range(len(liste)):
        if liste[i] < valS:
            minS = i
            valS = liste[i]

    return minS


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

    def relachement(sommet1: str, sommet2: str) -> bool:
        """
        Relachement d'un sommet.
        Cf. https://fr.wikipedia.org/wiki/Algorithme_de_Bellman
        :param sommet1: sommet 1
        :param sommet2: sommet 2
        :return: Vrai si relachêment, sinon false.
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
    listeArrets = [sommetArr]
    while sommetArr != arret_dep:
        sommetArr = distances_precedents[sommetArr][1]
        listeArrets.append(sommetArr)

    listeArrets = [arret_arriv] + listeArrets
    listeArrets.reverse()

    print("Algorithme de Bellman :")
    print(
            f"Pour aller de {arret_dep} à {arret_arriv}, il y a {round(distances_precedents[arret_arriv][0])} et il faut passer par les arrêts {listeArrets}.")

    return listeArrets, round(distances_precedents[arret_arriv][0])


# bellman("STLE", "BRNM")


def djikstra(arret_dep, arret_arriv):
    """
    Renvoie la distance la plus courte entre deux arrêts grâce à l'algorithme de Belmann.
    Cf. https://fr.wikipedia.org/wiki/Algorithme_de_Dijkstra

    :param arret_dep: arret de départ
    :type arret_dep: str
    :param arret_arriv: arret d'arrivée
    :type arret_arriv: str
    :return: une liste d'arrêts, la distance minimum
    :rtype: list, int
    """

    # Initialisation de la liste des distances
    sommet = indice_som(arret_dep)

    dist = [float('inf')] * len(noms_arrets)
    liste = [float('inf')] * len(noms_arrets)
    pred = [float('inf')] * len(noms_arrets)
    a_traiter = [i for i in range(len(noms_arrets))]

    # Afin d'éviter de passer par le sommet de départ, on l'enlève de la liste des sommets à traiter.
    a_traiter.remove(indice_som(arret_dep))
    pred[sommet] = sommet
    dist[sommet] = 0

    while len(a_traiter) != 0:
        for i in range(len(poids_bus)):
            if i in a_traiter:
                liste[i] = (poids_bus[sommet][i])

        for i in range(len(liste)):
            
            if liste[i] < float('inf'):
                if dist[i] > (dist[sommet] + liste[i]):
                    pred[i] = sommet
                    dist[i] = dist[sommet] + liste[i]
        for i in range(len(poids_bus)):
            liste[i] = (float('inf'))
        for i in a_traiter:
            liste[i] = dist[i]

        print(sommet)
        sommet = extract_min(liste)
        a_traiter.remove(sommet)
        

    chemin = []
    sommet = indice_som(arret_arriv)

    # Remontée afin d'avoir tous les sommets du chemin
    while sommet != indice_som(arret_dep):
        chemin.append(nom(sommet))
        sommet = pred[sommet]

    chemin.append(arret_dep)
    chemin.reverse()

    print(
            f"Pour aller de {arret_dep} à {arret_arriv}, il y a {round(dist[indice_som(arret_arriv)])}m et il faut passer par les arrêts {chemin}.")
    return chemin, round(dist[indice_som(arret_arriv)])


# djikstra("STLE", "BRNM")

def floyd_warshall(arret_dep, arret_arriv):
    """
    Renvoie la distance la plus courte entre deux arrêts grâce à l'algorithme de Floyd-Warshall.
    :param arret_dep: arret de départ
    :type arret_dep: str
    :param arret_arriv: arret d'arrivée
    :type arret_arriv: str
    :return: une liste d'arrêts, la distance minimum
    :rtype: list, int
    """
    # Création de la matrice Mk
    Mk = [[(0 if i == j else float("inf")) for j in range(len(noms_arrets))] for i in range(len(noms_arrets))]

    # Initialisation de la matrice Mk
    for i in range(len(noms_arrets)):
        for j in voisin(noms_arrets[i]):
            Mk[indice_som(noms_arrets[i])][indice_som(j)] = poids_bus[indice_som(noms_arrets[i])][indice_som(j)]

    # Création de la matrice Pk
    Pk = [[None for _ in range(len(noms_arrets))] for _ in range(len(noms_arrets))]

    # #Initialisation de la matrice Pk
    for i in range(len(noms_arrets)):
        for j in voisin(noms_arrets[i]):
            Pk[indice_som(j)][indice_som(noms_arrets[i])] = noms_arrets[i]

    # #Boucle de Floyd-Warshall
    for k in range(len(noms_arrets)):
        colonnes = [i for i in range(len(noms_arrets)) if i != k and Mk[i][k] != float("inf")]

        # Creation de la liste lignes
        lignes = [i for i in range(len(noms_arrets)) if i != k and Mk[k][i] != float("inf")]

        # Boucle de calcul de Mk
        for i in colonnes:
            for j in lignes:
                if Mk[i][k] + Mk[k][j] < Mk[i][j]:
                    Mk[i][j] = Mk[i][k] + Mk[k][j]
                    Pk[i][j] = Pk[i][k]

    # #Création de la liste des arrêts parcourus
    parcours = []
    arret_fin = arret_arriv
    parcours.append(arret_fin)

    while Pk[indice_som(arret_fin)][indice_som(arret_dep)] is not None:
        parcours.append(Pk[indice_som(arret_fin)][indice_som(arret_dep)])
        arret_fin = Pk[indice_som(arret_fin)][indice_som(arret_dep)]

    parcours.reverse()
    print("Algorithme de Floyd Warshall :")
    print(
            f"Pour aller de {arret_dep} à {arret_arriv}, il y a {round(Mk[indice_som(arret_arriv)][indice_som(arret_dep)])}m et il faut passer par les arrêts {parcours}.")
    return parcours, round(Mk[indice_som(arret_arriv)][indice_som(arret_dep)])


# floyd_warshall("STLE", "BRNM")

# Formule de l'approximation heuristique avec la méthode de la distance euclidienne
def calculHeuristique(arret_courant, arret_arriv):
    # Formule de l'heuristique euclidienne
    diffLongitude = abs(longitude(arret_courant) - longitude(arret_arriv))
    diffLattitude = abs(lattitude(arret_courant) - lattitude(arret_arriv))
    return sqrt(diffLongitude ** 2 + diffLattitude ** 2)


"""
On crée une classe arretAstar qui va créer un objet
contenant le nom de l'arret, son parent et les valeurs g,h et f nécessaire
pour l'algorithme Astar  
"""


class arretAstar:
    def __init__(self, nom = None, parent = None):
        self.nom = nom
        self.parent = parent

        self.g = 0
        self.h = 0
        self.f = 0


def astar(arret_dep, arret_arriv):
    # Creer les arrets de départ et de fin
    arretDep = arretAstar(arret_dep, None)
    arretFin = arretAstar(arret_arriv, None)

    # Initialiser les arrets de départ et de fin
    arretDep.g = arretDep.h = arretDep.f = 0
    arretFin.g = arretFin.h = arretFin.f = 0

    # Initialiser la liste ouverte et fermée
    l_ouverte = []
    l_fermee = []

    # Ajouter l'arret de depart dans la liste ouverte
    l_ouverte.append(arretDep)

    while len(l_ouverte) != 0:

        # Récuperer l'arret courant
        arret_courant = l_ouverte[0]
        index_courant = 0
        for index, arret in enumerate(l_ouverte):
            if arret.f < arret_courant.f:
                arret_courant = arret
                index_courant = index

        # Enlever l'arret courant de la liste ouverte et le mettre dans la liste fermée
        l_ouverte.pop(index_courant)
        l_fermee.append(arret_courant)

        # Génerer le parcours si on est arrivé a l'arret d'arrivé
        if arret_courant.nom == arret_arriv:
            parcours = []
            arretActuel = arret_courant
            while arretActuel is not None:
                parcours.append(arretActuel.nom)
                arretActuel = arretActuel.parent
            parcours = parcours[::-1]  # inversion de la liste pour l'avoir dans le bon sens

            arretParDefaut = arret_dep
            dist = 0

            for i in range(1, len(parcours)):
                dist += distance_arrets(arretParDefaut, parcours[i])
                arretParDefaut = parcours[i]

            print("Algortihme AStar (ou A étoile) :")
            print(
                    f"Pour aller de {arret_dep} à {arret_arriv}, il y a {round(dist)} mètres et il faut passer par les arrêts {parcours}.")
            return parcours, round(dist)

            # Creer les arret voisins
        arret_voisin = []
        voisins = voisin(arret_courant.nom)
        for nomArret in voisins:
            nouvel_arret = arretAstar(nomArret, arret_courant)
            arret_voisin.append(nouvel_arret)

        """
        Pour chaque voisin de l'arret courant, on regarde si:
            -on ne l'a pas deja parcouru
            -si on n'a pas deja un autre chemin menant à lui 
                plus rapide que celui qu'on regarde
        Et on ajoute le voisin dans la liste des arret à regarder par la suite
        """

        for arretV in arret_voisin:

            for elmt in l_fermee:
                if elmt == arretV:
                    continue

            arretV.g = arret_courant.g + distance_arrets(arretV.nom, arret_courant.nom)
            arretV.h = calculHeuristique(arretV.nom, arret_courant.nom)
            arretV.f = arretV.g + arretV.h

            for elmt in l_ouverte:
                if elmt == arretV:
                    continue
            l_ouverte.append(arretV)


# astar("STLE", "BRNM")



longitudeParArret = []
lattitudeParArret = []
for arret in donneesBus:
    longitudeParArret.append(longitude(arret))
    lattitudeParArret.append(lattitude(arret))

longitudeMax = max(longitudeParArret)
longitudeMin = min(longitudeParArret)

lattitudeMax = max(lattitudeParArret)
lattitudeMin = min(lattitudeParArret)


def bellmanGraphique(arret_dep, arret_arriv,window):

    # Initialisation de la liste des distances
    distances_precedents = {sommet: [float('inf'), None] for sommet in noms_arrets}
    # Pour le sommet de départ, on met la distance à 0.
    distances_precedents[arret_dep][0] = 0

    def relachement(sommet1: str, sommet2: str) -> bool:
        """
        Relachement d'un sommet.
        Cf. https://fr.wikipedia.org/wiki/Algorithme_de_Bellman
        :param sommet1: sommet 1
        :param sommet2: sommet 2
        :return: Vrai si relachêment, sinon false.
        """
        if distances_precedents[sommet1][0] + distance_arc(sommet1, sommet2) < distances_precedents[sommet2][0]:
            distances_precedents[sommet2][0] = distances_precedents[sommet1][0] + distance_arc(sommet1, sommet2)
            distances_precedents[sommet2][1] = sommet1
            gr.Line(gr.Point(*gpsToPixels(longitude(sommet1), lattitude(sommet1))),gr.Point(*gpsToPixels(longitude(sommet2), lattitude(sommet2)))).draw(window).setOutline("blue")
            sleep(0.05)
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
    listeArrets = [sommetArr]
    while sommetArr != arret_dep:
        sommetArr = distances_precedents[sommetArr][1]
        listeArrets.append(sommetArr)

    listeArrets = [arret_arriv] + listeArrets
    listeArrets.reverse()

    print("Arriver")
    for i in range(1,len(listeArrets)):
        sleep(0.5)
        gr.Line(gr.Point(*gpsToPixels(longitude(listeArrets[i-1]), lattitude(listeArrets[i-1]))),gr.Point(*gpsToPixels(longitude(listeArrets[i]), lattitude(listeArrets[i])))).draw(window).setOutline("green")
        

def djikstraGraphique(arret_dep, arret_arriv,window):
    """
    Renvoie la distance la plus courte entre deux arrêts grâce à l'algorithme de Belmann.
    Cf. https://fr.wikipedia.org/wiki/Algorithme_de_Dijkstra

    :param arret_dep: arret de départ
    :type arret_dep: str
    :param arret_arriv: arret d'arrivée
    :type arret_arriv: str
    :return: une liste d'arrêts, la distance minimum
    :rtype: list, int
    """

    # Initialisation de la liste des distances
    sommet = indice_som(arret_dep)

    dist = [float('inf')] * len(noms_arrets)
    liste = [float('inf')] * len(noms_arrets)
    pred = [float('inf')] * len(noms_arrets)
    a_traiter = [i for i in range(len(noms_arrets))]

    # Afin d'éviter de passer par le sommet de départ, on l'enlève de la liste des sommets à traiter.
    a_traiter.remove(indice_som(arret_dep))
    pred[sommet] = sommet
    dist[sommet] = 0

    while len(a_traiter) != 0:
        for i in range(len(poids_bus)):
            if i in a_traiter:
                liste[i] = (poids_bus[sommet][i])

        for i in range(len(liste)):
            if liste[i] < float('inf'):
                if dist[i] > (dist[sommet] + liste[i]):
                    gr.Line(gr.Point(*gpsToPixels(longitude(nom(sommet)), lattitude(nom(sommet)))), gr.Point(*gpsToPixels(longitude(nom(pred[sommet])), lattitude(nom(pred[sommet]))))).draw(window).setOutline("blue")
                    #sleep(0.05)
                    pred[i] = sommet
                    dist[i] = dist[sommet] + liste[i]
        for i in range(len(poids_bus)):
            liste[i] = (float('inf'))
        for i in a_traiter:
            liste[i] = dist[i]
        sommet = extract_min(liste)
        a_traiter.remove(sommet)

    chemin = []
    sommet = indice_som(arret_arriv)

    # Remontée afin d'avoir tous les sommets du chemin
    while sommet != indice_som(arret_dep):
        chemin.append(nom(sommet))
        sommet = pred[sommet]

    chemin.append(arret_dep)
    chemin.reverse()

    for i in range(1,len(chemin)):
        gr.Line(gr.Point(*gpsToPixels(longitude(chemin[i-1]), lattitude(chemin[i-1]))),gr.Point(*gpsToPixels(longitude(chemin[i]), lattitude(chemin[i])))).draw(window).setOutline("green")
        sleep(0.5)

def floyd_warshallGraphique(arret_dep, arret_arriv, window):
    """
    Renvoie la distance la plus courte entre deux arrêts grâce à l'algorithme de Floyd-Warshall.
    :param arret_dep: arret de départ
    :type arret_dep: str
    :param arret_arriv: arret d'arrivée
    :type arret_arriv: str
    :return: une liste d'arrêts, la distance minimum
    :rtype: list, int
    """
    # Création de la matrice Mk
    Mk = [[(0 if i == j else float("inf")) for j in range(len(noms_arrets))] for i in range(len(noms_arrets))]

    # Initialisation de la matrice Mk
    for i in range(len(noms_arrets)):
        for j in voisin(noms_arrets[i]):
            Mk[indice_som(noms_arrets[i])][indice_som(j)] = poids_bus[indice_som(noms_arrets[i])][indice_som(j)]

    # Création de la matrice Pk
    Pk = [[None for _ in range(len(noms_arrets))] for _ in range(len(noms_arrets))]

    # #Initialisation de la matrice Pk
    for i in range(len(noms_arrets)):
        for j in voisin(noms_arrets[i]):
            Pk[indice_som(j)][indice_som(noms_arrets[i])] = noms_arrets[i]

    # #Boucle de Floyd-Warshall
    for k in range(len(noms_arrets)):
        colonnes = [i for i in range(len(noms_arrets)) if i != k and Mk[i][k] != float("inf")]

        # Creation de la liste lignes
        lignes = [i for i in range(len(noms_arrets)) if i != k and Mk[k][i] != float("inf")]

        # Boucle de calcul de Mk
        for i in colonnes:
            for j in lignes:
                if Mk[i][k] + Mk[k][j] < Mk[i][j]:
                    Mk[i][j] = Mk[i][k] + Mk[k][j]
                    Pk[i][j] = Pk[i][k]

    # #Création de la liste des arrêts parcourus
    parcours = []
    arret_fin = arret_arriv
    parcours.append(arret_fin)

    while Pk[indice_som(arret_fin)][indice_som(arret_dep)] is not None:
        parcours.append(Pk[indice_som(arret_fin)][indice_som(arret_dep)])
        arret_fin = Pk[indice_som(arret_fin)][indice_som(arret_dep)]

    parcours.reverse()
    
    for i in range(1,len(parcours)):
        sleep(0.5)
        gr.Line(gr.Point(*gpsToPixels(longitude(parcours[i-1]), lattitude(parcours[i-1]))),gr.Point(*gpsToPixels(longitude(parcours[i]), lattitude(parcours[i])))).draw(window).setOutline("green")
    
    
class Point:
    def __init__(self, nom: str, lat: float, long: float, voisins: []):
        self.nom = nom
        self.lat = lat
        self.long = long
        self.voisins = [arret for arret in voisins]

    def __repr__(self):
        return f"\"{self.nom}\" ({self.lat}, {self.long}) -> [{self.voisins}]"

    def getVoisins(self):
        return self.voisins

    def getNom(self):
        return self.nom

    def getLat(self):
        return self.lat

    def getLong(self):
        return self.long

    def getCoords(self):
        return self.lat, self.long



tousPoints = [Point(arret,  *donneesBus[arret]) for arret in donneesBus]


def testGraphique():
    win = gr.GraphWin("Test", LARGEUR_FENETRE, HAUTEUR_FENETRE)

    diffLat = lattitudeMax - lattitudeMin + 0.05
    diffLong = longitudeMax - longitudeMin + 0.1

    ratio = diffLong / diffLat

    def gpsToPixels(x, y):
        return LARGEUR_FENETRE * abs(longitudeMin - x) / diffLong * ratio,\
               HAUTEUR_FENETRE - (HAUTEUR_FENETRE * abs(lattitudeMin - y)) / diffLat

    for point in tousPoints:
        gr.Circle(gr.Point(*gpsToPixels(point.getLong(), point.getLat())), 4).draw(win).setOutline("red")

        #voisin = point.voisins[0]

        #gr.Circle(gr.Point(*gpsToPixels(voisin.getLong(), voisin.getLat())), 4).draw(win).setOutline("red")

       # break


#     for i in range(len(longitudeParArret)):

    # for j in range(len(longitudeParArret)):
    #     gr.Line(
    #             gr.Point((10 + (lattitudeParArret[i] - lattitudeMin) * 3000),
    #                      600 - (lattitudeParArret[i] - lattitudeMin) * 3000),
    #             gr.Point((10 + (lattitudeParArret[j] - lattitudeMin) * 3000),
    #                      600 - (lattitudeParArret[j] - lattitudeMin) * 3000)
    #             ).draw(win)
    #     for i in range(len(longitudeParArret)):
    #         long = 10 + (longitudeParArret[i] - longitudeMin) * 3000
    #         lat = 600 - (lattitudeParArret[i] - lattitudeMin) * 3000
    #         # print(long, lat)
    #
    #         gr.Circle(gr.Point(lat, long), 4).draw(win).setOutline("red")
    win.getMouse()
    win.close()

#testGraphique()

diffLat = lattitudeMax - lattitudeMin + 0.05
diffLong = longitudeMax - longitudeMin + 0.1

ratio = diffLong / diffLat

def gpsToPixels(x, y):
        return LARGEUR_FENETRE * abs(longitudeMin - x) / diffLong * ratio,\
               HAUTEUR_FENETRE - (HAUTEUR_FENETRE * abs(lattitudeMin - y)) / diffLat
               
def testPointEtTrait():
    
    win = gr.GraphWin("Test", LARGEUR_FENETRE, HAUTEUR_FENETRE)

    for point in tousPoints:
        gr.Circle(gr.Point(*gpsToPixels(point.getLong(), point.getLat())), 4).draw(win).setOutline("red")
        for voisinArret in point.getVoisins():
            gr.Line(gr.Point(*gpsToPixels(point.getLong(), point.getLat())),gr.Point(*gpsToPixels(longitude(voisinArret), lattitude(voisinArret)))).draw(win)
    #bellmanGraphique("STLE", "BRNM",win)
    #djikstraGraphique("STLE", "BRNM",win)
    floyd_warshallGraphique("STLE", "BRNM",win)
    win.getMouse()
    win.close()

testPointEtTrait()

