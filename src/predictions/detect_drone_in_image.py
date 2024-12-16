

import cv2
from src.config import model
from src.utils.draw_bounding import draw_boxes


def detect_drone_in_image(image_path):

    outputs = {
        "predictions": []
    }

    resultat = model(image_path)
    try:
        if len(resultat) > 0:
            #Recuperation de la listes des coordonnées des objets detectés
            xywh_list = resultat[0].boxes.xywh.tolist()

            #recuperation de la list des confiances de prédiction
            conf_list = resultat[0].boxes.conf.tolist()

            for i in range(len(conf_list)):
                output = {
                    'x': int(xywh_list[i][0]),
                    'y': int(xywh_list[i][1]),
                    'w': int(xywh_list[i][2]),
                    'h': int(xywh_list[i][3]),
                    'confidence': round(conf_list[i],2),
                    'class': "drone"
                }
                outputs["predictions"].append(output)
                # Creation d'identifiant pour chaque objet
                identities = [int(i + 1) for i in range(len(conf_list))]

                image_detect = draw_boxes(img=cv2.imread(image_path), bbox=resultat[0].boxes.xyxy.tolist(),
                                   identities=identities, offset=(0, 0))
                """# Vérification si l'image a été correctement chargée
                if image is not None:
                    image_file_name = os.path.basename(image_path)
                    save_dir_path = f"{os.path.abspath('runs')}/{image_file_name}"
                    #Enregistrement de l'image
                    cv2.imwrite(save_dir_path, image) """

            return outputs, image_detect, len(conf_list)
        else:
            return None
    except Exception as e:
        print("Unable to open the video:", e)
        return None



if __name__ == '__main__':
 predict,image, nbr = detect_drone_in_image(image_path="E:/AllProject/dronetrackingapi/src/uploadfolder/image.png")
 cv2.imshow("Image", image)
 cv2.waitKey(0)  # Attendre une touche pour fermer la fenêtre
 cv2.destroyAllWindows()
 print(predict)




