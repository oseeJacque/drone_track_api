from collections import deque
from random import random

import cv2
import numpy as np

data_deque = {}

def xyxy_to_xywh(*xyxy):
    """
    Calculates the relative bounding box from absolute pixel values.

    Arguments:
    *xyxy -- Les coordonnées xyxy de la boîte englobante

    Returns:
    Un tuple contenant les coordonnées relatives de la boîte englobante (x_c, y_c, w, h)
    """

    # Calcul de la coordonnée x la plus petite entre les coins supérieur gauche et inférieur droit
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])

    # Calcul de la coordonnée y la plus petite entre les coins supérieur gauche et inférieur droit
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])

    # Calcul de la largeur absolue de la boîte englobante
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())

    # Calcul de la hauteur absolue de la boîte englobante
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())

    # Calcul de la coordonnée x du centre de la boîte englobante
    x_c = (bbox_left + bbox_w / 2)

    # Calcul de la coordonnée y du centre de la boîte englobante
    y_c = (bbox_top + bbox_h / 2)

    # Attribution de la largeur de la boîte englobante
    w = bbox_w

    # Attribution de la hauteur de la boîte englobante
    h = bbox_h

    # Retourne les coordonnées relatives de la boîte englobante sous forme de tuple (x_c, y_c, w, h)
    return x_c, y_c, w, h


def draw_border(img, pt1, pt2, color, thickness, r, d):
    """
       Dessine une bordure avec coins arrondis autour d'une région rectangulaire définie par les points pt1 et pt2.

       Args:
           img (numpy.ndarray): L'image sur laquelle dessiner la bordure.
           pt1 (tuple): Les coordonnées (x, y) du premier point définissant le coin supérieur gauche du rectangle.
           pt2 (tuple): Les coordonnées (x, y) du deuxième point définissant le coin inférieur droit du rectangle.
           color (tuple): La couleur de la bordure au format (B, G, R).
           thickness (int): L'épaisseur de la bordure.
           r (int): Le rayon des coins arrondis de la bordure.
           d (int): La longueur des lignes horizontales et verticales des coins arrondis.

       Returns:
           numpy.ndarray: L'image avec la bordure dessinée.
       """
    x1, y1 = pt1
    x2, y2 = pt2

    # Définition des coordonnées des points du rectangle
    # Coin supérieur gauche
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Coin supérieur droit
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Coin inférieur gauche
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Coin inférieur droit
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    # Dessin des rectangles internes
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    # Dessin des cercles aux coins du rectangle
    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    # Retourne l'image modifiée
    return img

def xyxy_to_tlwh(bbox_xyxy):
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (top, left, width, height) format.

    Arguments:
    bbox_xyxy -- List of bounding boxes in (x1, y1, x2, y2) format.

    Returns:
    List of bounding boxes in (top, left, width, height) format.
    """

    # Liste pour stocker les boîtes englobantes au format (top, left, width, height)
    tlwh_bboxs = []

    # Parcours de chaque boîte englobante dans la liste bbox_xyxy
    for i, box in enumerate(bbox_xyxy):
        # Extraction des coordonnées x1, y1, x2, y2 de la boîte englobante
        x1, y1, x2, y2 = [int(i) for i in box]

        # Calcul de la coordonnée top (supérieure) de la boîte englobante
        top = x1

        # Calcul de la coordonnée left (gauche) de la boîte englobante
        left = y1

        # Calcul de la largeur de la boîte englobante
        w = int(x2 - x1)

        # Calcul de la hauteur de la boîte englobante
        h = int(y2 - y1)

        # Création d'un objet tlwh (top, left, width, height) pour la boîte englobante
        tlwh_obj = [top, left, w, h]

        # Ajout de l'objet tlwh à la liste des boîtes englobantes
        tlwh_bboxs.append(tlwh_obj)

    # Retourne la liste des boîtes englobantes au format (top, left, width, height)
    return tlwh_bboxs

def draw_border(img, pt1, pt2, color, thickness, r, d):
    """
       Dessine une bordure avec coins arrondis autour d'une région rectangulaire définie par les points pt1 et pt2.

       Args:
           img (numpy.ndarray): L'image sur laquelle dessiner la bordure.
           pt1 (tuple): Les coordonnées (x, y) du premier point définissant le coin supérieur gauche du rectangle.
           pt2 (tuple): Les coordonnées (x, y) du deuxième point définissant le coin inférieur droit du rectangle.
           color (tuple): La couleur de la bordure au format (B, G, R).
           thickness (int): L'épaisseur de la bordure.
           r (int): Le rayon des coins arrondis de la bordure.
           d (int): La longueur des lignes horizontales et verticales des coins arrondis.

       Returns:
           numpy.ndarray: L'image avec la bordure dessinée.
       """
    x1, y1 = pt1
    x2, y2 = pt2

    # Définition des coordonnées des points du rectangle
    # Coin supérieur gauche
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Coin supérieur droit
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Coin inférieur gauche
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Coin inférieur droit
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    # Dessin des rectangles internes
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    # Dessin des cercles aux coins du rectangle
    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    # Retourne l'image modifiée
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    """
    Trace une boîte englobante avec une étiquette sur une image.

    Args:
        x (list): Les coordonnées de la boîte englobante [x1, y1, x2, y2].
        img (numpy.ndarray): L'image sur laquelle la boîte englobante doit être tracée.
        color (tuple, optional): La couleur de la boîte englobante au format (R, G, B). Si non spécifié, une couleur aléatoire sera utilisée.
        label (str, optional): L'étiquette à afficher à côté de la boîte englobante. Si non spécifié, aucune étiquette ne sera affichée.
        line_thickness (int, optional): L'épaisseur de la ligne de la boîte englobante. Si non spécifié, une valeur par défaut sera utilisée.

    Returns:
        numpy.ndarray: L'image avec la boîte englobante et l'étiquette dessinées.
    """

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    # Définition de la couleur de la boîte (si non spécifiée, une couleur aléatoire est utilisée)
    color = color or [random.randint(0, 255) for _ in range(3)]

    # Coordonnées des coins de la boîte englobante
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    # Dessin de la boîte englobante sur l'image
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # Épaisseur de la police de caractères

        # Calcul de la taille du texte
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        # Dessin du contour autour de l'étiquette
        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)

        # Ajout du texte de l'étiquette à l'image
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    """
    Dessine les boîtes englobantes avec les étiquettes et les trajectoires sur une image.

    Arguments :
        img (numpy.ndarray) : L'image sur laquelle les boîtes englobantes doivent être dessinées.
        bbox (list) : Les coordonnées des boîtes englobantes des objets détectés.
        object_id (list) : Les identifiants des objets détectés.
        identities (list, optionnel) : Les identifiants des objets.
        offset (tuple, optionnel) : Le décalage pour les coordonnées des boîtes englobantes.

    Retour :
        numpy.ndarray : L'image modifiée avec les boîtes englobantes, les étiquettes et les trajectoires dessinées.
    """

    height, width, _ = img.shape

    # Supprimer les points de suivi du tampon si l'objet est perdu
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    # Parcourir les boîtes englobantes et les noms des objets détectés
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Calcul du centre du bord inférieur de la boîte
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # Obtention de l'ID de l'objet
        id = int(identities[i]) if identities is not None else 0

        # Création d'un nouveau tampon pour un nouvel objet
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)

        # Calcul de la couleur en fonction de l'ID de l'objet
        color = (0, 149, 255)  # Couleur bleu


        # Obtention du nom de l'objet
        obj_name = "Drone"

        # Création de l'étiquette avec l'ID et le nom de l'objet
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        # Ajout du centre au tampon
        data_deque[id].appendleft(center)

        # Dessin de la boîte englobante avec l'étiquette sur l'image
        UI_box(box, img, label=label, color=color, line_thickness=2)

        # Dessin de la trajectoire
        for i in range(1, len(data_deque[id])):
            # Vérification si une valeur du tampon est nulle
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue

            # Calcul de l'épaisseur dynamique de la trajectoire
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)

            # Dessin de la trajectoire
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

    return img