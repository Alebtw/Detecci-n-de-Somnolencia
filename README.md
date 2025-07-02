
# Sistema de Detección de Somnolencia con Visión por Computadora y Deep Learning

Este proyecto implementa un sistema de detección de somnolencia en tiempo real utilizando técnicas de visión artificial y redes neuronales profundas. La solución analiza expresiones faciales y señales fisiológicas relacionadas con la fatiga, con el objetivo de prevenir accidentes en contextos como la conducción o el manejo de maquinaria.

## 🧠 ¿Cómo funciona?

- Captura video en tiempo real desde una cámara web.
- Detecta el rostro y los ojos del usuario.
- Analiza la frecuencia de parpadeo, cierre ocular prolongado y señales de fatiga.
- Clasifica el estado del usuario mediante una red neuronal convolucional personalizada.
- Emite alertas visuales y/o sonoras si detecta somnolencia.

## 🛠 Tecnologías utilizadas

- Python 3.x  
- OpenCV  
- TensorFlow / Keras  
- NumPy  
- dlib (opcional para landmarks faciales)  
- Matplotlib (para pruebas y visualización)  

## ⚙️ Requisitos

Instala los requerimientos con:

```bash
pip install -r requirements.txt
opencv-python==4.9.0.80
tensorflow==2.15.0
keras==2.15.0
numpy==1.26.4
matplotlib==3.8.4
dlib==19.24.2
imutils==0.5.4
pillow==10.3.0
scipy==1.13.1
