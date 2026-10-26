"""Seq2Seq mínimo: Encoder LSTM, Decoder LSTM y lectura lineal compartida."""
import torch
from torch import nn


class Seq2Seq(nn.Module):
    """Resume la historia con un Encoder y genera 24 valores con un Decoder.

    Ambos son LSTM de una capa. El contexto es el par de estados finales (h,c),
    que transporta lo aprendido de la historia hacia la generación futura.
    """
    def __init__(self, entradas, ocultas=16, calendario_dim=6):
        """Define pesos para F entradas históricas, H estados y seis calendarios.

        El Decoder recibe un valor anterior y el calendario de la hora futura.
        Linear convierte cada estado de H componentes en un objetivo escalar.
        Las capas se crean una vez y comparten sus pesos entre pasos temporales.
        """
        super().__init__()
        self.encoder = nn.LSTM(entradas, ocultas, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(1 + calendario_dim, ocultas, num_layers=1, batch_first=True)
        self.salida = nn.Linear(ocultas, 1)

    def forward(self, historia, calendario_futuro):
        """Forward Pass: transforma (B,48,F) y (B,24,6) en (B,24) normalizado.

        B es la cantidad de ventanas del minilote; los estados de cada ventana
        son independientes. Los objetivos reales no se reciben en este método:
        sirven después para calcular la pérdida, nunca como entrada futura.
        """
        # historia: (B, 48, F); h,c finales: (1, B, 16).
        assert calendario_futuro.shape[1:] == (24, 6)
        # La omisión de estados iniciales hace que PyTorch comience h,c en cero.
        # Al terminar la historia, ambos estados inicializan el Decoder.
        _, estados = self.encoder(historia)
        anterior = historia[:, -1, 0:1]  # Último objetivo conocido (o imputado causalmente).
        predicciones = []
        for k in range(24):
            # Entrada (B,7): valor anterior y seis señales de calendario conocidas.
            # unsqueeze añade el eje temporal: un paso de forma (B,1,7).
            entrada = torch.cat([anterior, calendario_futuro[:, k, :]], dim=-1)
            secuencia, estados = self.decoder(entrada.unsqueeze(1), estados)
            # Los h,c cambiaron; los pesos siguen siendo los mismos. La lectura
            # lineal produce el siguiente valor y este se realimenta sin detach.
            anterior = self.salida(secuencia[:, 0, :])  # (B, 1)
            predicciones.append(anterior)
        # Mismos pesos en los 24 pasos; h,c cambian. No detach, no Teacher Forcing.
        # Los objetivos reales no forman parte de la firma de forward.
        return torch.cat(predicciones, dim=1)  # (B, 24)


def cargar_modelo(ruta):
    """Reconstruye la arquitectura y pesos del checkpoint; devuelve red y respaldo.

    eval() prepara la red para inferencia. No se entrena ni se elige otra época:
    se recupera la que minimizó MSE normalizado de Validation.
    """
    ckpt = torch.load(ruta, map_location='cpu', weights_only=True)
    modelo = Seq2Seq(**ckpt['arquitectura'])
    modelo.load_state_dict(ckpt['pesos'])
    modelo.eval()
    return modelo, ckpt
