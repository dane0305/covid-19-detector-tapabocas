# Detector de tapabocas con OpenCV y Keras/Tensorflow
![](demo.gif)

Puedes ver toda la explicación del código en mi blog: [RaspiBlog](http://blog.joseb.co)

## Modo de uso:

Instalar librerías y dependencias necesarias:

```
pip install -r requirements.txt
```

Entrenamiento:

```
cd /path_al_repo/
python train_mask_detector.py --dataset dataset
```

Detectar tapabocas en imágenes:

```
cd /path_al_repo/
python detect_mask_image.py --image examples/example_01.png
```

Detectar tapabocas con webcams:

```
cd /path_al_repo/
python detect_mask_video.py
```