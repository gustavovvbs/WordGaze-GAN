import torch 
import torch.nn.functional as F
from torch import nn

# NORMALIZANDO AS COORDENADAS

# tensor = torch.randn(128, 3)

# coordinates = tensor[:, :2]

# min_values = coordinates.min(dim=0).values
# max_values = coordinates.max(dim=0).values

# normalized_coordinates = (coordinates - min_values) / (max_values - min_values) * 2 - 1

# normalized_tensor = tensor.clone()
# normalized_tensor[:, :2] = normalized_coordinates

class Encoder(nn.Module):

    def __init__(self, input_dim=384, zdim=32):  ##deduzi q o zdim eh [1, 32] pq la ele repete o noise pelo tamanho do input
                                            #ai p bater a dimensao de 35, teria q ser [1, 32](z), pq qnd fica repetido pelo espaco do input fica[128, 32] concatenando com o input q eh c ctz [128, 3], fica [128, 35]
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 112), #inputdim=128x3=384
            nn.LeakyReLU(0.01),
            nn.Linear(112, 96),
            nn.LeakyReLU(0.01),
            nn.Linear(96, 48),
            nn.LeakyReLU(0.01),
            nn.Linear(48, zdim),  
            nn.LeakyReLU(0.01),
        )

        self.z_mean = nn.Linear(zdim, zdim) 
        self.z_log_var = nn.Linear(zdim, zdim) 

    def reparameterize(self, z_mu, z_log_var):
        epsilon = torch.randn_like(z_log_var).to(z_log_var.device)
        z = z_mu + epsilon * torch.exp(z_log_var/2)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded, z_mean, z_log_var


if __name__ == "__main__":
    model = Encoder()  
    print("Output dimension:", model(torch.randn(1, 384))[0].shape)  #ver se a dimensao ta batendo