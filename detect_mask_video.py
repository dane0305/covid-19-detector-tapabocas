# Modo de uso:
# python detect_mask_video.py

# importamos las librerías necesarias
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# guardamos las dimensiones del frame recibido y lo
	# convertimos a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pasamos el frame por el detector de rostros y lo guardamos
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# inizializamos la lista de rostros
	# sus localizacion en la imagen y su predicción
	faces = []
	locs = []
	preds = []

	# hacemos un loop en las detecciones
	for i in range(0, detections.shape[2]):
		# extraemos la probabilidad asociadad
		# con la detección
		confidence = detections[0, 0, i, 2]

		# filtramos las detecciones menores a nuestro confidence que 
		# fijamos anteriormente: 50%
		if confidence > args["confidence"]:
			# calculamos las coordenadas x,y del objeto detectado
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# nos aseguramos que esté dentro del frame
			# que le pasamos
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extraemos el ROI en este caso el rostro y convertimos de BRG a RGB
			# lo reducimos a 224x224 y lo procesamos
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# agregamos el bounding box y el rostro detectado en sus respectivas
			# listas
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# solo hacer detecciones si al menos 1 rostro es detectado
	if len(faces) > 0:
		# para una rápida inferencia procesamos todas los rostros al mismo tiempo
		preds = maskNet.predict(faces)

	# retornamos las predicciones y las ubicaciones del rostro
	return (locs, preds)

# construimos el argumento
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# cargamos el detector de rostros
print("[INFO] cargando el modelo de detector de rostros 🙂...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# cargamos el detector de tapabocas
print("[INFO] cargando el modelo de detector de tapabocas 😷...")
maskNet = load_model(args["model"])

# inicializamos el script y seteamos un warm up para la cámara
print("[INFO] iniciando video stream... 🍻")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# hacemos un loop por frames del video
while True:
	# guardamos cada frame del video y luego lo reducimos a 800x800 de tamaño
	frame = vs.read()
	frame = imutils.resize(frame, width=800)

	# detectamos rostros con nuestra función
	# y detectamos si tiene tapabocas o no
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# hacemos un loop sobre los rostros detectados y sus ubicaciones
	for (box, pred) in zip(locs, preds):
		# llamamos al bounding box y la predicción
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determinamos la clase  y seteamos un color
		# en el bbox
		label = "Con Tapabocas" if mask > withoutMask else "Sin Tapabocas"
		color = (0, 255, 0) if label == "Con Tapabocas" else (0, 0, 255)

		# incluímos la probabilidad
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# mostramos la etiqueta o label y el bounding box en cada frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# mostramos la salida
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# si presionamos q salimos
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()