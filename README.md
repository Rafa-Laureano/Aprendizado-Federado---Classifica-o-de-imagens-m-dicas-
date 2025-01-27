# Aprendizado-Federado---Classifica-o-de-imagens-m-dicas-
Uso do aprendizado federado para treinamento em arquiteturas open source para a área médica. O aprendizado federado nesse projeto é realizado com a framework Flower. Nesse projeto também há o uso do FastAI, pytorch e outros. 

A implementação prática utilizou dois dispositivos Raspberry Pi 5 como clientes e um notebook configurado como servidor.

Para trocar o modelo de treinamento nos "Clientes" basta trocar o modelo, em "model" na função:

def create_model():
    model = resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: positivo e negativo
    return model
