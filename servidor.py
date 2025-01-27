import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
 
# Função de agregação para métricas de treinamento
def fit_metrics_aggregation_fn(metrics):
    accuracies = [m[1]["accuracy"] for m in metrics]  # Acesse o segundo elemento da tupla
    return {"accuracy": sum(accuracies) / len(accuracies)}
 
# Função de agregação para métricas de avaliação
def evaluate_metrics_aggregation_fn(metrics):
    losses = [m[1]["loss"] for m in metrics]  # Acesse o segundo elemento da tupla
    accuracies = [m[1]["accuracy"] for m in metrics]
    return {"loss": sum(losses) / len(losses), "accuracy": sum(accuracies) / len(accuracies)}
 
# Definir a estratégia personalizada com funções de agregação de métricas
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
)
 
if __name__ == "__main__":
    # Configuração do servidor Flower com agregação de métricas
    config = ServerConfig(num_rounds=3)
 
    # Iniciar o servidor Flower com a estratégia personalizada
    fl.server.start_server(
        server_address="192.168.60.107:8080",
        config=config,
        strategy=strategy,
    )
