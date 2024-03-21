# importer les bibliothèques nécessaires
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    # calculer les distances euclidiennes entre les deux ensembles de
    # points de repère verticaux des yeux (coordonnées x, y)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # calculer la distance euclidienne entre les points de repère horizontaux
    # des yeux (coordonnées x, y)
    C = dist.euclidean(eye[0], eye[3])
    # calculer le rapport d'aspect des yeux
    ear = (A + B) / (2.0 * C)
    # retourner le rapport d'aspect des yeux
    return ear

# construire l'analyseur d'arguments et analyser les arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="chemin du prédicteur de points de repère faciaux")
ap.add_argument("-v", "--video", type=str, default="",
    help="chemin du fichier vidéo en entrée")
args = vars(ap.parse_args())

# définir deux constantes, une pour le rapport d'aspect des yeux indiquant
# un clignement d'œil et une deuxième constante pour le nombre de trames consécutives
# que l'œil doit être en dessous du seuil
SEUIL_RAPPORT_ASPECT = 0.3
TRAMES_CONSECUTIVES_SEUIL = 3

# initialiser le temps écoulé depuis que les yeux sont fermés
temps_yeux_fermes = 0
SEUIL_TEMPS_YEUX_FERMES = 2  # seuil de temps pour l'alerte (en secondes)

# initialiser les compteurs de trames et le nombre total de clignements
COMPTEUR = 0
TOTAL = 0

# initialiser le détecteur de visages de dlib (basé sur HOG) et créer
# le prédicteur de points de repère faciaux
print("[INFO] Chargement du prédicteur de points de repère faciaux...")
detecteur = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# obtenir les indices des points de repère faciaux pour l'œil gauche et
# droit, respectivement
(debutGauche, finGauche) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(debutDroit, finDroit) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# démarrer le flux vidéo
print("[INFO] Démarrage du flux vidéo...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
time.sleep(1.0)

# boucle sur les trames du flux vidéo
while True:
    # si c'est un flux vidéo à partir d'un fichier, alors nous devons vérifier s'il y a
    # d'autres trames à traiter dans le tampon
    if fileStream and not vs.more():
        break

    # récupérer la trame du flux vidéo, la redimensionner
    # et la convertir en niveaux de gris
    frame = vs.read()
    if frame is not None:
        frame = imutils.resize(frame, width=900)
    else:
        print("Erreur : la trame est None.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # détecter les visages dans la trame en niveaux de gris
    rects = detecteur(gray, 0)

    # boucle sur les détections de visage
    for rect in rects:
        # déterminer les points de repère faciaux pour la région du visage, puis
        # convertir les coordonnées des points de repère faciaux (x, y) en un tableau NumPy
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extraire les coordonnées des yeux gauche et droit, puis utiliser
        # les coordonnées pour calculer le rapport d'aspect des yeux pour les deux yeux
        oeilGauche = shape[debutGauche:finGauche]
        oeilDroit = shape[debutDroit:finDroit]
        rapportAspectOeilGauche = eye_aspect_ratio(oeilGauche)
        rapportAspectOeilDroit = eye_aspect_ratio(oeilDroit)

        # moyenne du rapport d'aspect des yeux pour les deux yeux
        rapportAspectOeil = (rapportAspectOeilGauche + rapportAspectOeilDroit) / 2.0

        # calculer l'enveloppe convexe pour l'œil gauche et droit, puis
        # visualiser chacun des yeux
        enveloppeConvexeOeilGauche = cv2.convexHull(oeilGauche)
        enveloppeConvexeOeilDroit = cv2.convexHull(oeilDroit)
        cv2.drawContours(frame, [enveloppeConvexeOeilGauche], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [enveloppeConvexeOeilDroit], -1, (0, 255, 0), 1)

        # vérifier si le rapport d'aspect des yeux est en dessous du seuil de clignement
        # et si c'est le cas, incrémenter le compteur de trames de clignotement
        if rapportAspectOeil < SEUIL_RAPPORT_ASPECT:
            COMPTEUR += 1
        # sinon, le rapport d'aspect des yeux n'est pas en dessous du seuil de clignement
        else:
            # si les yeux étaient fermés, mais ne le sont plus, réinitialiser le temps
            temps_yeux_fermes = 0

            # si les yeux étaient fermés pendant un nombre suffisant de
            # trames, alors incrémenter le nombre total de clignotements
            if COMPTEUR >= TRAMES_CONSECUTIVES_SEUIL:
                TOTAL += 1
            # réinitialiser le compteur de trames de clignotement
            COMPTEUR = 0

        # si les yeux sont fermés, incrémenter le temps écoulé
        if rapportAspectOeil < SEUIL_RAPPORT_ASPECT:
            temps_yeux_fermes += 1
        else:
            # si les yeux étaient fermés, mais ne le sont plus, réinitialiser le temps
            temps_yeux_fermes = 0

        # si les yeux sont fermés depuis plus de 3 secondes, afficher une alerte
        if temps_yeux_fermes >= (SEUIL_TEMPS_YEUX_FERMES * 30):  # 30 images par seconde
            cv2.putText(frame, "ALERT: Risque de fatigue oculaire!    ", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # dessiner le nombre total de clignotements sur la trame avec
        # le rapport d'aspect des yeux calculé pour la trame
        cv2.putText(frame, "Clignotements: {} ".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Rapport d'Aspect : {:.2f}    ".format(rapportAspectOeil), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # afficher la trame
    cv2.imshow("Trame", frame)
    key = cv2.waitKey(1) & 0xFF

    # si la touche `q` est enfoncée, sortir de la boucle
    if key == ord("q"):
        break

# nettoyer un peu
cv2.destroyAllWindows()
vs.stop()
