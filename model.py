import torch 
from torch import nn
from torch.nn.utils import spectral_norm

"""    
    Encoder block: encoda o dado real num vetor de mu e log_var 
"""

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(128*3, 192),
            nn.LeakyReLU(0.2),
            nn.Linear(192, 96),
            nn.LeakyReLU(),
            nn.Linear(96, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 32),
            nn.LeakyReLU()
        )

        self.mu = nn.Linear(32, 32)
        self.log_var = nn.Linear(32, 32)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Achata o input para entrada na rede
        x = self.encoder(x)
        mu = self.mu(x)  # Média da distribuição latente
        log_var = self.log_var(x)  # Logaritmo da variância da distribuição latente
        return mu, log_var

""" Discriminador: classifica se o dado é real ou falso """

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Rede neural do discriminador com normalização espectral
        self.discriminator = nn.Sequential(
            spectral_norm(nn.Linear(128*3, 192)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(192, 96)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(96, 48)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(48, 24)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(24, 1))
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        activations = []

        for layer in self.discriminator:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations.append(x)
                
        return x, activations

""" Gerador: gera um dado falso (dominio B) a partir de um vetor de ruído concatenado com o prototipo (dominio A)"""

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        # Rede LSTM para gerar sequências de gestos
        self.lstm1 = nn.LSTM(z_dim + 3, z_dim, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(2*z_dim, z_dim, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(2*z_dim, z_dim, bidirectional=True, batch_first=True)
        self.lstm4 = nn.LSTM(2*z_dim, z_dim, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(2*z_dim, 3)

    def forward(self, x, z):
        z = z.unsqueeze(dim = 1).expand(-1, x.size(1), -1)

        x = torch.cat([x, z], dim=-1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.dense(x)
        return torch.tanh(x)  # Garante que a saída está entre -1 e 1


    def get_activations(self, x, z):
        z = z.unsqueeze(dim = 1).expand(-1, x.size(1), -1)

        x = torch.cat([x, z], dim=-1)

        x, _ = self.lstm1(x)
        x1 = self.lstm2(x1)
        x2 = self.lstm3(x2)
        x3 = self.lstm4(x3)
        
        return x


