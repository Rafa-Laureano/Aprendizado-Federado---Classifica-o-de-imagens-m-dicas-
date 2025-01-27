import flwr as fl
from fastai.vision.all import *
import torch
import torch.nn as nn
import csv
import os
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Função para carregar os dados
def load_data():
    path = Path("/home/rafaella/Desktop/federado/pneumonia/training")
    dls = ImageDataLoaders.from_folder(
        path,
        valid_pct=0.2,
        bs=16,  # Reduzir o batch size para diminuir o uso de memória
        item_tfms=Resize(224),
        num_workers=0  # Define num_workers=0 para evitar multiprocessing
    )
    return dls

# Callback personalizado para salvar métricas e pesos em cada época
class SaveCallback(Callback):
    def __init__(self, save_path="training_logs"):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def after_epoch(self):
        # Verificar se as métricas existem e estão disponíveis
        if self.learn.recorder.values and len(self.learn.recorder.values[-1]) >= 3:
            metrics = {
                "epoch": self.epoch,
                "accuracy": self.learn.recorder.values[-1][2],  # Acurácia
                "loss": self.learn.recorder.values[-1][0],     # Perda
            }
        else:
            metrics = {
                "epoch": self.epoch,
                "accuracy": None,
                "loss": None,
            }

        # Salvar métricas da época
        metrics_file = os.path.join(self.save_path, f"metrics_epoch_{self.epoch}.csv")
        with open(metrics_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)

        # Salvar pesos da época
        weights_file = os.path.join(self.save_path, f"weights_epoch_{self.epoch}.pth")
        torch.save(self.learn.model.state_dict(), weights_file)

# Função para salvar métricas em CSV (usado fora do callback, se necessário)
def save_metrics(metrics, filename):
    os.makedirs("metrics", exist_ok=True)  # Garante que a pasta exista
    filepath = os.path.join("metrics", filename)
    with open(filepath, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})

# Função para salvar pesos do modelo (usado fora do callback, se necessário)
def save_weights(model, filename):
    os.makedirs("weights", exist_ok=True)  # Garante que a pasta exista
    filepath = os.path.join("weights", filename)
    torch.save(model.state_dict(), filepath)

# Definir o modelo EfficientNet-B0 com a cabeça personalizada
def create_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features  # Número de features na última camada
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # Atualizar para 2 classes
    return model

# Classe do Cliente Flower
class PneumoniaClient(fl.client.NumPyClient):
    def __init__(self):
        self.dls = load_data()
        self.model = create_model()
        self.learn = Learner(
            self.dls,
            self.model,
            loss_func=CrossEntropyLossFlat(),
            metrics=accuracy,
            cbs=SaveCallback(save_path="training_logs"),  # Adiciona o callback
        )
        
        self.round_counter = 1  # Contador local para as rodadas

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for val, param in zip(parameters, self.model.parameters()):
            param.data = torch.tensor(val).to(param.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.learn.fine_tune(6)  # Define 5 épocas para o treinamento
        # Salvar métricas e pesos finais da rodada
        metrics = {"accuracy": self.learn.recorder.values[-1][2]} if self.learn.recorder.values else {"accuracy": None}
        save_metrics(metrics, f"fit_metrics_client_round_{self.round_counter}.csv")
        save_weights(self.model, f"model_weights_client_fit_round_{self.round_counter}.pth")

        self.round_counter += 1  # Incrementa o contador local
        return self.get_parameters(config={}), len(self.dls.train), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.learn.validate()

        # Salvar métricas e pesos da avaliação
        metrics = {"loss": float(loss), "accuracy": float(accuracy)}
        save_metrics(metrics, f"eval_metrics_client_round_{self.round_counter}.csv")
        save_weights(self.model, f"model_weights_client_eval_round_{self.round_counter}.pth")

        self.round_counter += 1  # Incrementa o contador local
        return float(loss), len(self.dls.valid), metrics

# Iniciar o cliente Flower
if __name__ == "__main__":
    client = PneumoniaClient()
    fl.client.start_client(
        server_address="192.168.60.107:8080",
        client=client,
    )
