# Aprendizado-Federado---Classifica-o-de-imagens-m-dicas-
Uso do aprendizado federado para treinamento em arquiteturas open source para a área médica. O aprendizado federado nesse projeto é realizado com a framework Flower. Nesse projeto também há o uso do FastAI, pytorch e outros. 

A implementação prática utilizou dois dispositivos Raspberry Pi 5 como clientes e um notebook configurado como servidor.

Para trocar o modelo usado para treinamento nos "Clientes" basta trocar o modelo, em "model" na função "def create_model():"

exemplo:

model = *modelo*(weights="IMAGENET1K_V1")

O dataset usado se encontra na plataforma Kaggle "https://www.kaggle.com/datasets/adnanalaref/pneumonia-chest-xray".

os códigos "cliente1", podem ser usado para cliente2, 3, assim por diante.

