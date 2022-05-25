# ~~~~~~~~ Import de la bibliothèque json pour traiter un fichier json ~~~~~~~ #
# ~~~~~~~~~~~~~~~ Import de sin, cos et acos de la bibliothèque ~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~ math pour la fonction distanceGPS ~~~~~~~~~~~~~~~~~~~~ #
import json
from math import sin, cos, acos

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   A : importer les donnees du fichier JSON dans un dictionnaire donneesbus   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

with open('Fichiers/donneesbus.json') as json_file:
    donneesbus = json.load(json_file)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#         B : Créer une liste noms_arrets contenants le noms des arrêts        #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

noms_arrets=list(donneesbus.keys())
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                            C : Créer les fonctions                           #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def nom(ind):
    """Retourne le nom de l'arrêt à l'indice ind

    :param ind: indice de l'arrêt
    :type ind: int
    :return: un nom
    :rtype: str
    """
    return noms_arrets[ind]


def indice_som(nom_som):
    """Retourne l'indice de l'arrêt de nom nom_som

    :param nom_som: nom de l'arrêt
    :type nom_som: str
    :return: un indice
    :rtype: int
    """
    return noms_arrets.index(nom_som)

def latitude(nom_som):
    """Retourne la latitude de l'arrêt de nom nom_som

    :param nom_som: nom de l'arrêt
    :type nom_som: str
    :return: une latitude
    :rtype: float
    """
    return donneesbus[nom_som][0]

def longitude(nom_som):
    """Retourne la longitude de l'arrêt de nom nom_som

    :param nom_som: nom de l'arrêt
    :type nom_som: str
    :return: une longitude
    :rtype: float
    """
    return donneesbus[nom_som][1]

def voisin(nom_som):
    """Retourne la liste des voisins de l'arrêt de nom nom_som

    :param nom_som: nom de l'arrêt
    :type nom_som: str
    :return: une liste de noms d'arrêts
    :rtype: list
    """
    return donneesbus[nom_som][2]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#            D : Le réseau du bus peut être modélisé par des graphes           #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                              Par un dictionnaire                             #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ~~~~~~~~~~~~~~~~~~~~~~~ Création du dictionnaire vide ~~~~~~~~~~~~~~~~~~~~~~ #
dic_bus = {}
for arret in noms_arrets:
    # ~~~~~~~~~~~~~~~~~~~~ Pour chaque arret dans noms_arrets ~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~ On crée une liste vide qui a pour clé le nom de l'arrêt ~~~~~~~~~ #
    dic_bus[arret] = []
    for arret2 in noms_arrets:
        # ~~~~~~~~~~~~~ Pour chaque arret2 dans noms_arrets ~~~~~~~~~~~~~~~~~~~~~ #
        if arret2 in voisin(arret):
            # ~~~~~~~~~~~~~~~ Si arret2 est dans la liste des voisins ~~~~~~~~~~~~~ #
            dic_bus[arret].append(arret2)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                            Par une liste de liste                            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ~~~~~~~~~~~~~~~~~~~~~~~~~ Création d'une liste vide ~~~~~~~~~~~~~~~~~~~~~~~~ #
mat_bus = []
for arret in noms_arrets:
    # ~~~~~~~~~~~~~~~~~~~~ Pour chaque arret dans noms_arrets ~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~ On crée une liste vide ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    liste = []
    for arret2 in noms_arrets:
        # ~~~~~~~~~~~~~ Pour chaque arret2 dans noms_arrets ~~~~~~~~~~~~~~~~~~~~~ #
        if arret2 in voisin(arret):
            # ~~~~~~~~~~~~~~~ Si arret2 est dans la liste des voisins ~~~~~~~~~~~~~ #
            liste.append(1)
        else:
            # ~~~~~~~~~~~~~~~ Si arret2 n'est pas dans la liste des voisins ~~~~~~~~~~~~~~~ #
            liste.append(0)
    # ~~~~~~~~~~~~~~~~~ Ajouter la liste liste à la liste mat_bus ~~~~~~~~~~~~~~~~ #
    mat_bus.append(liste)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                 E : Distance                                 #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def distanceGPS(latA,latB,longA,longB): 
    # Conversions des latitudes en radians 
    ltA=latA/180*3.14
    ltB=latB/180*3.14
    loA=longA/180*3.14
    loB=longB/180*3.14
    # Rayon de la terre en mètres (sphère IAG-GRS80) 
    RT = 6378137 
    # angle en radians entre les 2 points  
    S = acos(round(sin(ltA)*sin(ltB) + cos(ltA)*cos(ltB)*cos(abs(loB-loA)),14)) 
    # distance entre les 2 points, comptée sur un arc de grand cercle 
    return S*RT

def distarrets(arret1, arret2):
    """Retourne la distance entre les arrêts arret1 et arret2

    :param arret1 : nom de l'arrêt 1
    :type : str
    :param arret2 : nom de l'arrêt 2
    :type : str
    :return: distance entre les deux arrêts
    :rtype: float
    """
    return(round(distanceGPS(latitude(arret1),latitude(arret2),longitude(arret1),longitude(arret2))))

def distarc(arret1, arret2):
    """Retourne la distance entre les arrêts arret1 et arret2 s'il existe sinon retourne 'inf'

    :param arret1 : nom de l'arrêt 1
    :type : str
    :param arret2 : nom de l'arrêt 2
    :type : str
    :return: distance entre les deux arrêts
    :rtype: float
    """
    if arret2 in voisin(arret1):
        return(distarrets(arret1, arret2))
    else:
        return float("inf")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                    F : Modélisation par un graphe pondéré                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ~~~~~~~~~ Matrice des poids (liste de listes), poids_bus prenant en ~~~~~~~~ #
# ~~~~~~~~~~~ compte à la fois l’existence des arcs et leur poids. ~~~~~~~~~~~ #

poids_bus = []
for arret in noms_arrets:
    # ~~ Pour chaque arret dans noms_arrets, on l'ajoute dans la liste poids_bus ~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~ Puis on crée une liste vide ~~~~~~~~~~~~~~~~~~~~~~~ #
    liste = []
    for arret2 in noms_arrets:
        # ~~~~~ On regarde si l'arret2 est dans la liste des voisins de l'arret1 ~~~~~ #
        if arret2 in voisin(arret):
            # ~~~~~ Si oui, on ajoute la distance entre les deux arrêts dans la liste ~~~~ #
            liste.append(distarc(arret, arret2))
        else:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sinon on ajoute 'inf' ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            liste.append(float("inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~ On ajoute la liste à la liste poids_bus ~~~~~~~~~~~~~~~~~~ #
    poids_bus.append(liste)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                    ETAPE 2                                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def extract_min(lst):
    """Retourne le sommet de poids minimum de la liste lst
    
    :param lst : liste des arrets
    :type lst: list
    :return: le sommet de poids minimum
    :rtype: int
    """
    
    minS = 100000
    valS = 100000
    for i in range (len(lst)):
        if (lst[i] < valS):
            minS = i
            valS = lst[i]
    return(minS)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                   DJIKSTRA                                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def Djikstra(arret_dep, arret_arriv):
    """Calcule le plus court chemin entre deux points arret_dep et arret_ariv
    en utilisant l'algorithme de Djikstra
    :param arret_dep: arret de départ
    :type arret_dep: str
    :param arret_arriv: arret d'arrivée
    :type arret_arriv: str
    :return: une liste d'arrêts, la distance minimum
    :rtype: list, int
    """
    # ~~~~~~~~~~~~~~~~ Déclaration et initialisation des variables ~~~~~~~~~~~~~~~ #
    dist=[]
    pred=[]
    a_traiter=[]
    lst=[]
    som=indice_som(arret_dep)
    compteur = 0

    for i in range(len(poids_bus)):
        dist.append(float('inf'))
        lst.append(float('inf'))
        pred.append(float('inf'))
        a_traiter.append(i)
    
    a_traiter.remove(indice_som(arret_dep)) # On enlève l'arret de départ de la liste des arrets à traiter
    pred[som] = som # On met le prédécesseur de l'arret de départ à lui-même
    dist[som] = 0 # On met la distance de l'arret de départ à 0
    
    # ~~~~~~~~~~~~~~~~ Boucle tant qu'il y a des sommets à traiter ~~~~~~~~~~~~~~~ #
    while(len(a_traiter) != 0):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initialisation liste ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        for i in range(len(poids_bus)):
            lst[i] = (float('inf'))
    # ~~~~~~~~~~~~~~~~ Les poids de chaque arc commencant par som ~~~~~~~~~~~~~~~~ #
        for i in range(len(poids_bus)):
            if(i in a_traiter):
                lst[i] = (poids_bus[som][i])
    
    # ~~~~~~~~ Comparaison de la dist du sommet avec la nouvelle distance ~~~~~~~~ #
        for i in range(len(lst)):
            if(lst[i] < float('inf')):
                if(dist[i] > (dist[som]+lst[i])):
                    pred[i] = som
                    dist[i]
                    dist[i] = dist[som]+lst[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Re initialisation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        for i in range(len(poids_bus)):
            lst[i] = (float('inf'))
        # ~~~~~~~~~~~~~~~~~~ La distance des poids encore à traiter ~~~~~~~~~~~~~~~~~~ #
        for i in (a_traiter):
            lst[i] = dist[i]
        # ~~~~~~~~~~~~~~~~ Initialisation du nouveau sommet a traiter ~~~~~~~~~~~~~~~~ #
        som = extract_min(lst)
        a_traiter.remove(som)
        compteur = compteur+1
        
    # ~~~~~~~~ Liste des arrets du chemin (avec arret_dep et arret_arriv) ~~~~~~~~ #
    chemin = []
    som = indice_som(arret_arriv)
    while som != indice_som(arret_dep):
        chemin.append(nom(som))
        som = pred[som]
    chemin.append(arret_dep)
    chemin.reverse()
    
    # Affichage des résultats
    print("Le chemin permettant d'aller de", arret_dep, "jusqu'à",arret_arriv, "est :",str(chemin).replace("[","").replace("'","").replace(", "," -> ").replace("]",""),"\nLe trajet comptera", len(chemin), "arrets." ,"\nLa distance entre ces 2 arrets est de :", dist[indice_som(arret_arriv)],"m.")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ APPEL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
arret_dep = input("Arret dep :")
arret_arriv = input("Arret fin :")
if(arret_dep not in noms_arrets):
    print("L'arret", arret_dep,"n'existe pas.")
    if(arret_arriv not in noms_arrets):
        print("L'arret", arret_arriv,"n'existe pas.")
else:
    Djikstra(arret_dep, arret_arriv)
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                   BELMANN                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def Belmann(arret_dep,arret_arriv):
    """Calcule le plus court chemin entre deux arrêts arret_dep et arret_arriv
    en utilisant l'algorithme de Bellman-Ford
    :param arret_dep: arret de départ
    :type arret_dep: str
    :param arret_arriv: arret d'arrivée
    :type arret_arriv: str
    :return: une liste d'arrêts, la distance minimum
    :rtype: list, int
    """
    #Création la liste des prédecesseurs
    pred = [None]*len(noms_arrets)
    #Création la liste des distances
    dist = [None]*len(noms_arrets)

    #Initialisation des listes
    for i in range(len(noms_arrets)):
        pred[i] = None
        dist[i] = float("inf")
    
    dist[indice_som(arret_dep)] = 0

    #Boucle de Bellman
    for i in range(len(noms_arrets)-1):
        for j in noms_arrets:
            for k in voisin(j):
                if dist[indice_som(j)] + poids_bus[indice_som(j)][indice_som(k)] < dist[indice_som(k)]:
                    dist[indice_som(k)] = dist[indice_som(j)] + poids_bus[indice_som(j)][indice_som(k)]
                    pred[indice_som(k)] = j
    
    #Création de la liste des arrêts parcourus
    arret_fin = arret_arriv
    chemin = []
    chemin.append(arret_fin)
    while pred[indice_som(arret_fin)] != None:
        chemin.append(pred[indice_som(arret_fin)])
        arret_fin = pred[indice_som(arret_fin)]
    chemin.reverse()
    
    print ("Le chemin permettant d'aller de", arret_dep, "jusqu'à",arret_arriv, "est :",str(chemin).replace("[","").replace("'","").replace(", "," -> ").replace("]",""),"\nLe trajet comptera", len(chemin), "arrets." ,"\nLa distance entre ces 2 arrets est de :", dist[indice_som(arret_arriv)],"m.")
"""
arret_dep = input("Arret dep :")
arret_arriv = input("Arret fin :")
if(arret_dep not in noms_arrets):
    print("L'arret", arret_dep,"n'existe pas.")
    if(arret_arriv not in noms_arrets):
        print("L'arret", arret_arriv,"n'existe pas.")
else:
    Belmann(arret_dep, arret_arriv)
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                FLOYD WARSHALL                                #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def FloydWarshall(arret_dep,arret_arriv):
    """Calcule le plus court chemin entre deux arrêts arret_dep et arret_arriv
    en utilisant l'algorithme de Floyd-Warshall
    :param arret_dep: arret de départ
    :type arret_dep: str
    :param arret_arriv: arret d'arrivée
    :type arret_arriv: str
    :return: une liste d'arrêts, la distance minimum
    :rtype: list, int
    """
    #Création de la matrice Mk
    Mk = []
    for i in range(len(noms_arrets)):
        Mk.append([])
        for j in range(len(noms_arrets)):
            if i == j:
                Mk[i].append(0)
            else:
                Mk[i].append(float("inf"))
    
    #Initialisation de la matrice Mk
    for i in range(len(noms_arrets)):
        for j in voisin(noms_arrets[i]):
            Mk[indice_som(noms_arrets[i])][indice_som(j)] = poids_bus[indice_som(noms_arrets[i])][indice_som(j)]
    #Création de la matrice Pk
    Pk = []
    for i in range(len(noms_arrets)):
        Pk.append([])
        for j in range(len(noms_arrets)):
            Pk[i].append(None)

    #Initialisation de la matrice Pk
    for i in range(len(noms_arrets)):
        for j in voisin(noms_arrets[i]):
            Pk[indice_som(j)][indice_som(noms_arrets[i])] = noms_arrets[i]
    
    #Boucle de Floyd-Warshall
    for k in range(len(noms_arrets)):
        #Création de la liste colonnes
        colonnes = []
        for i in range(len(noms_arrets)):
                if i != k and Mk[i][k] != float("inf"):
                    colonnes.append(i)          
            
        #Creation de la liste lignes
        lignes = []
        for i in range(len(noms_arrets)):
            if i != k and Mk[k][i] != float("inf"):
                lignes.append(i)
        
        #Boucle de calcul de Mk
        for i in colonnes:
            for j in lignes:
                if Mk[i][k] + Mk[k][j] < Mk[i][j]:
                    Mk[i][j] = Mk[i][k] + Mk[k][j]
                    Pk[i][j] = Pk[i][k]
        
    
    #Création de la liste des arrêts parcourus
    parcours = []
    arret_fin = arret_arriv
    parcours.append(arret_fin)
    while Pk[indice_som(arret_fin)][indice_som(arret_dep)] != None:
        parcours.append(Pk[indice_som(arret_fin)][indice_som(arret_dep)])
        arret_fin = Pk[indice_som(arret_fin)][indice_som(arret_dep)]
    parcours.reverse()
    print("Le chemin permettant d'aller de", arret_dep, "jusqu'à",arret_arriv, "est :",str(parcours).replace("[","").replace("'","").replace(", "," -> ").replace("]",""),"\nLe trajet comptera", len(parcours), "arrets." ,"\nLa distance entre ces 2 arrets est de :", Mk[indice_som(arret_arriv)][indice_som(arret_dep)],"m.")
"""
arret_dep = input("Arret dep :")
arret_arriv = input("Arret fin :")
if(arret_dep not in noms_arrets):
    print("L'arret", arret_dep,"n'existe pas.")
    if(arret_arriv not in noms_arrets):
        print("L'arret", arret_arriv,"n'existe pas.")
else:
    FloydWarshall(arret_dep, arret_arriv)
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                    A_STAR                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def A_star(arret_dep,arret_arriv):
    #A_star est une fonction qui prend en paramètre les deux arrêts de départ et d'arrivée.
    #Elle renvoie le plus court chemin, sous forme de la liste des arrêts parcourus ainsi, que la distance minimum.
    #Elle utilise la fonction heuristique ainsi que la fonction de recherche de plus court chemin.

    openList = [arret_dep]
    closedList = []
    pred = []
    dist = []

    #Initialisation de la liste des prédécesseurs
    for i in range(len(noms_arrets)):
        pred.append(None)
    #Initialisation de la liste des distances
    for i in range(len(noms_arrets)):
        if i == indice_som(arret_dep):
            dist.append(0)
        else:
            dist.append(float("inf"))
    #Initialisation de la liste des distances
    for i in range(len(noms_arrets)):
        dist[indice_som(noms_arrets[i])] = poids_bus[indice_som(noms_arrets[i])][indice_som(arret_dep)]
    #Boucle de A*
    #Si la file d'attente est vide, il n'y a aucun chemin du nœud initial au nœud d'arrivée, ce qui interrompt l'algorithme.
    while len(openList) != 0:

        #L'algorithme retire le premier nœud de la file d'attente prioritaire, le nœud courant.
        arret_courant = openList[0]

        #Si le nœud retenu est le nœud d'arrivée, A* reconstruit le chemin complet et s'arrête.
        if(arret_courant == arret_arriv):
            break
        
        #Si le nœud n'est pas le nœud d'arrivée, de nouveaux nœuds sont créés ;
        #Pour chaque nœud successif, A* calcule son coût et le stocke avec le nœud.
        #On cherche les nœuds qui sont accessibles par le nœud courant.
        for i in range(len(openList)):
            if dist[indice_som(openList[i])] < dist[indice_som(arret_courant)]:
                arret_courant = openList[i]
        #On le supprime de la liste des arrets à visiter
        openList.remove(arret_courant)
        #On l'ajoute à la liste des arrets visités
        closedList.append(arret_courant)
        #On récupère les voisins de l'arret courant
        voisins = voisin(arret_courant)
        #Boucle sur les voisins
        for i in voisins:
            #Si le voisin n'est pas dans la liste des arrets visités 
            #Si le voisin n'est pas dans la liste des arrets à visiter
            if i not in closedList and i not in openList:
                #On l'ajoute à la liste des arrets à visiter 
                openList.append(i)
            #On calcule la distance entre l'arret courant et le voisin
            dist_voisin = dist[indice_som(arret_courant)] + poids_bus[indice_som(arret_courant)][indice_som(i)]
            #Si la distance est inférieure à la distance actuelle
            if dist_voisin < dist[indice_som(i)]:
                #On met à jour la distance
                dist[indice_som(i)] = dist_voisin
                #On met à jour le prédécesseur
                pred[indice_som(i)] = arret_courant

    #Création de la liste des arrêts parcourus
    parcours = []
    arret_fin = arret_arriv

    parcours.append(arret_arriv)
    while pred[indice_som(arret_arriv)] != None:
        parcours.append(pred[indice_som(arret_arriv)])
        arret_arriv = pred[indice_som(arret_arriv)]
    parcours.append(arret_dep)
    parcours.reverse()

    print("A_Star : Le chemin permettant d'aller de", arret_dep, "jusqu'à",arret_fin, "est :",str(parcours).replace("[","").replace("'","").replace(", "," -> ").replace("]",""),"\nLe trajet comptera", len(parcours), "arrets." ,"\nLa distance entre ces 2 arrets est de :", dist[indice_som(arret_fin)],"m.")

"""
arret_dep = input("Arret dep :")
arret_arriv = input("Arret fin :")
if(arret_dep not in noms_arrets):
    print("L'arret", arret_dep,"n'existe pas.")
    if(arret_arriv not in noms_arrets):
        print("L'arret", arret_arriv,"n'existe pas.")
else:
    A_star(arret_dep, arret_arriv)
"""

print(A_star("STLE", "BRNM"))