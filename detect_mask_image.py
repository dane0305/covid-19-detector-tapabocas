# Modo de uso
# python detect_mask_image.py --image examples/example_01.png

# importamos los paquetes necesarios
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# construimos los argumentos que le vamos a pasar
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path donde se encuentra la imagen a detectar")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path donde se encuentra el modelo de detector de rostros")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Cargamos nuestro modelo de detector de rostros
print("[INFO] Cargamos modelo de detector de rostros 🙂...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Cargamos nuestro modelo previamente entrenado
print("[INFO] loading face mask detector model 🙂...")
model = load_model(args["model"])

# leemos la imagen para detectar y guardamos las dimensiones espaciales 
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# construimos un un blob de la imagen
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# pasamos el blob por el detector de rostros
print("[INFO] Detectando rostros 🙂...")
net.setInput(blob)
detections = net.forward()

# hacemos un loop en las detecciones para detectar todos los rostros
for i in range(0, detections.shape[2]):
	# extraemos la probabilidad de las detecciones
	confidence = detections[0, 0, i, 2]
	# filtramos las detecciones bajas configuradas previamente
	# en este caso descargar las que son menores al 50%
	if confidence > args["confidence"]:
		# procesamos las coordenadas  (x, y)- del bounding box del objeto detectado
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# nos aseguramos que el bbox está dentros de las dimensiones del 
		# frame
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# extramos el ROI en este caso del rostro
		# lo reducirmos a 224x224 y lo procesamos
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# pasamos la variable face (rostro detectado) por el modelo de tapabocas
		# para saber si tiene o no
		(mask, withoutMask) = model.predict(face)[0]

		# determinamos un label y coloreamos el bounding box (recuadro)
		# de acuerdo a la detección
		label = "Tapabocas" if mask > withoutMask else "Sin Tapabocas"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# incluímos la probabilidad en la etiqueta o label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# mostramos el label y el bbox en cada imagen detectada
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)