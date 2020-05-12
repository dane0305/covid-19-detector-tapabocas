# Modo de uso
# python train_mask_detector.py --dataset dataset

# importamos los paquetes necesarios
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construimos los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# Inicializamos el learning rate inicial, numeros de epochs,
# y el batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Precargamos los datasets de la carpeta donde tenemos nuestras imágenes
print("[INFO] Cargando imágenes...🖼️")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Creamos un loope en directorio de las imágenes
for imagePath in imagePaths:
	# extraemos el nombre o la clase del nombre del archivo
	label = imagePath.split(os.path.sep)[-2]

	# cargamos la imagen en una resolución (224x224) y los procesamos
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# actualizamos los arrays data y labels
	data.append(image)
	labels.append(label)

# los convertimos en numpyarrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# binarizamos los datos de las etiquetas
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# particionamos los datos con un 75% de
# de los datos para training y un 25% para testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# seteamos un dataGenerator para hacer el data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Cargamos el modelo de MobileNetv2, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# contruímos el modelo y la función de activación y el tamaño del pool
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# El principal modelo FC (fully connected) será nuestro headModel 
model = Model(inputs=baseModel.input, outputs=headModel)

# hacer un loop en todoos los layers y detenerlos en caso de 
# no estar actualizandose en el training
for layer in baseModel.layers:
	layer.trainable = False

# Compilamos el modelo
print("[INFO] compilando modelo...🧠")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# entrenamos el modelo 
print("[INFO] Entrenando el modelo...💪")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Realizamos un test con los datos de testing ya con nuestro modelo entrenado
print("[INFO] Evaluando el modelo...📈")
predIdxs = model.predict(testX, batch_size=BS)

# guardamos las probabilidades de los imagenes del set de testing con su correspondiente label o
# etiqueta
predIdxs = np.argmax(predIdxs, axis=1)

# mostramos un reporte 
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# guardamos el modelo en formato h5
print("[INFO] Guardando el modelo...💾")
model.save(args["model"], save_format="h5")

# imprimimos la curva del loss
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss y Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])