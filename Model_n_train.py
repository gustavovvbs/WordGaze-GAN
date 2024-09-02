import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
import numpy as np
import os
import torch.nn.utils.spectral_norm as spectral_norm
import matplotlib.pyplot as plt
import time
import io

from PIL import Image


from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import normalize


import torch
import numpy as np

# Criar o gráfico
def plot_sequence(real_paths, fake_data, word, batch_count):
    plt.figure()
    
    # Plotar a sequência real
    real_x = real_paths[0, :, 0].cpu().detach().numpy()
    real_y = real_paths[0, :, 1].cpu().detach().numpy()
    print(len(real_x))
    # plt.plot(real_x, real_y, label='Real Sequence', marker='o')
    
    # normalized_fake = normalize(fake_data, dim=1)
    # fake_data = normalized_fake
    # fake_data = fake_data*2 - 1

    # Plotar a sequência gerada
    fake_x = fake_data[0, :, 0].cpu().detach().numpy()
    fake_y = fake_data[0, :, 1].cpu().detach().numpy()
    plt.plot(fake_x, fake_y, label='Generated Sequence', marker='x')

    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    plt.title(f'Sequence Plot for Word "{word}" at Batch {batch_count}')
    plt.legend()

    # Salvar o plot em um buffer de memória
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Converter o buffer em uma imagem
    image = Image.open(buf)
    image = np.array(image)
    
    # Adicionar a imagem ao TensorBoard
    writer.add_image(f'Sequence Plot/{word}_batch_{batch_count}', image, batch_count, dataformats='HWC')

    # Fechar a plotagem para liberar memória
    plt.close()


keyboard = {
    'q': (0.05, 0.16666666666666669), 'w': (0.15, 0.16666666666666669), 'e': (0.25, 0.16666666666666669),
    'r': (0.35, 0.16666666666666669), 't': (0.45, 0.16666666666666669), 'y': (0.55, 0.16666666666666669),
    'u': (0.65, 0.16666666666666669), 'i': (0.75, 0.16666666666666669), 'o': (0.85, 0.16666666666666669),
    'p': (0.95, 0.16666666666666669), 
    
    'a': (0.1, 0.5), 's': (0.2, 0.5), 'd': (0.3, 0.5),
    'f': (0.4, 0.5), 'g': (0.5, 0.5), 'h': (0.6, 0.5), 'j': (0.7, 0.5), 'k': (0.8, 0.5),
    'l': (0.9, 0.5), 

    'z': (0.2, 0.8333333333333334), 'x': (0.3, 0.8333333333333334),
    'c': (0.4, 0.8333333333333334), 'v': (0.5, 0.8333333333333334), 'b': (0.6, 0.8333333333333334),
    'n': (0.7, 0.8333333333333334), 'm': (0.8, 0.8333333333333334)
}

def transform_coordinates(x, y):
    """ Apply the basis transformation to the coordinates. """
    x_new = x * 2 - 1
    y_new = -y * 2 + 1
    return x_new, y_new

def get_coordinates(sentence):
    coordinates = []

    for char in sentence:
        if char in keyboard:
            x, y = keyboard[char]
            x_transformed, y_transformed = transform_coordinates(x, y)
            coordinates.append([x_transformed, y_transformed])
            
    return coordinates

def get_word_prototype(word):
    try:
        coordinates = get_coordinates(word)

        x, y = zip(*coordinates)
        x, y = np.array(x), np.array(y)

        n = 128
        k = len(coordinates)

        path_points = []

        for i in range(len(coordinates) - 1):
            num_points = (n - k) // (k - 1)  # Calculate the number of points between each pair of coordinates

            x_values = np.linspace(coordinates[i][0], coordinates[i+1][0], num_points, endpoint=False) 
            y_values = np.linspace(coordinates[i][1], coordinates[i+1][1], num_points, endpoint=False)

            path_points.extend(list(zip(x_values, y_values)))
        
        path_points.append(coordinates[-1])

        if len(path_points) < 128:
            while len(path_points) < 128:
                path_points.append(coordinates[-1])
        elif len(path_points) > 128:
            path_points = path_points[:128]

        path_pointsnp = np.array(path_points)

        ones = np.zeros((128, 1))
        path_pointsnp = np.hstack((path_pointsnp, ones))

        # # Plot the word path
        # plt.figure(figsize=(10, 6))
        # plt.plot(path_pointsnp[:, 0], path_pointsnp[:, 1], marker='o')
        # plt.xlim(-2, 2)
        # plt.ylim(-2, 2)
        # plt.title(f"Path for word: {word}")
        # plt.show()

        return torch.tensor(path_pointsnp, dtype=torch.float32)
    except Exception as e:
        print(f'Error with word: {word}: {e}')
        return torch.zeros((128, 3), dtype=torch.float32)

def get_batch_prototypes(words):
    prototypes = []
    for word in words:
        prototypes.append(get_word_prototype(word.lower()))
    
    return torch.stack(prototypes)


###<------------DATA------------>###
# Carrega dados do arquivo JSON
with open('gestures_data.json', 'r') as f:
    data = json.load(f)

def generate_dataset():
    for item in data:
        yield item

# Converte os dados pra uma lista de dicionários PyTorch
dataset = list(generate_dataset())
dataset = [{'word': item['word'], 'path': torch.tensor(item['path'])} for item in dataset]

BATCH_SIZE = 512
learning_rate = 1e-4
DISC_UPDATES = 5
lambda_feat = 1
lambda_rec = 5
lambda_lat = 0.5
lambda_KLD = 0.05
EPOCHS = 10


####<---GRADIENT PENALTY TEST-----> [[[[NOT IMPLEMENTED YET]]]]####
def gradient_penalty(discriminator, generator, real_data, fake_data):
    batch_size, c, h, w = real_data.shape
    eps = torch.randn((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated_image = eps*real_data + (1 - eps)*fake_data

    interpolated_score = discriminator(interpolated_image)
    interpolated_gradients = torch.autograd.grad(
        inputs = interpolated_image,
        outputs = interpolated_score,
        grad_outputs = torch.ones_like(interpolated_score),
        create_graph = True,
        retain_graph = True
    )[0]

    interpolated_gradients = interpolated_gradients.view(interpolated_gradients.shape[0], -1)
    gradients_norm = interpolated_gradients.norm(2, dim = 1)
    penalty_term = torch.mean((gradients_norm - 1)**2)
    return penalty_term 

###<--------VARIATIONAL ENCODER------->###

class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        # Rede neural para codificação
        self.encoder = nn.Sequential(
            nn.Linear(128*3, 192),
            nn.LeakyReLU(),
            nn.Linear(192, 96),
            nn.LeakyReLU(),
            nn.Linear(96, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 32),
            nn.LeakyReLU()
        )
        self.mu = nn.Linear(32, 32)  # Cálculo da média
        self.log_var = nn.Linear(32, 32)  # Cálculo da variância log

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Achata o input para entrada na rede
        x = self.encoder(x)
        mu = self.mu(x)  # Média da distribuição latente
        log_var = self.log_var(x)  # Logaritmo da variância da distribuição latente
        return mu, log_var

    def reparameterize(self, mu, log_var):
        eps = torch.randn_like(mu)  # Ruído normal para reparametrização
        log_var = torch.clamp(log_var, min=-5, max=5)
        return eps * torch.exp(log_var * 0.5) + mu  # Reparametrização para amostragem do espaço latente

    def L_lat(self, z, z_generated):
        # Perda latente: mede a diferença entre o código latente original e o gerado
        return torch.mean(torch.abs(z - z_generated))

    def L_KLD(self, mu, log_var):
      kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var + 9e-9), dim=1)
      return kld.mean()

###<------------DISCRIMINATOR------------>###

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
        return self.discriminator(x)

    def disc_loss(self, fake_data, real_data):
        # Perda do discriminador: diferença média entre saídas de dados reais e falsificados
        fake_output = self(fake_data)
        real_output = self(real_data)
        return torch.mean(fake_output) - torch.mean(real_output)

    def extract_features(self, x):
        # Extração de características intermediárias para calcular a perda de características
        features = []
        x = x.view(x.size(0), -1)
        for layer in self.discriminator[:-1]:
            x = layer(x)
            features.append(x)
        return features

###<------------GENERATOR------------>###

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Rede LSTM para gerar sequências de gestos
        self.lstm1 = nn.LSTM(35, 32, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(64, 32, bidirectional=True, batch_first=True)
        self.lstm4 = nn.LSTM(64, 32, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(64, 3)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.dense(x)
        return torch.tanh(x)  # Garante que a saída está entre -1 e 1

    def gen_loss(self, discriminator, encoder, fake_data, real_data, z, z_generated):
        # Perda do gerador: combinação da perda do discriminador, perda de características, perda latente e perda de reconstrução
        fake_output = discriminator(fake_data)
        real_output = discriminator(real_data)
        disc_loss = -discriminator.disc_loss(fake_data, real_data)
        feat_loss = self.L_feat(discriminator, fake_data, real_data)
        rec_loss = self.L_rec(fake_data, real_data)
        lat_loss = encoder.L_lat(z, z_generated)
        mu_fake, log_var_fake = encoder(fake_data)
        kld_loss = encoder.L_KLD(mu_fake, log_var_fake)
        loss_G = -disc_loss + lambda_feat * feat_loss + lambda_rec * rec_loss + lambda_lat * lat_loss + lambda_KLD * kld_loss
        return loss_G

    def L_feat(self, discriminator, fake_output, real_output):
        # Perda de características: diferença entre as características extraídas dos dados gerados e reais
        loss = 0
        fake_features = discriminator.extract_features(fake_output)
        real_features = discriminator.extract_features(real_output)
        for fake_feature, real_feature in zip(fake_features, real_features):
            loss += torch.mean(torch.abs(fake_feature - real_feature))
        return loss

    def L_rec(self, fake_output, real_output):
        # Perda de reconstrução: diferença entre a saída gerada e o dado real
        return torch.mean(torch.abs(fake_output - real_output))

###<-<------------TRAINING------------>###

class MODEL:
    def __init__(self, generator, discriminator1, discriminator2, encoder):
        self.generator = generator
        self.discriminator1 = discriminator1
        self.discriminator2 = discriminator2
        self.encoder = encoder
        # Otimizadores para o gerador e discriminadores
        self.generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
        self.discriminator_optimizer1 = optim.Adam(discriminator1.parameters(), lr=learning_rate)
        self.discriminator_optimizer2 = optim.Adam(discriminator2.parameters(), lr=learning_rate)

    def train_step(self, real_data):
      with torch.autograd.detect_anomaly():
        real_path = real_data['path']
        batch_size = real_path.size(0)
        z = torch.randn(batch_size, 32)  # Amostras aleatórias para o espaço latente

        # Treinar o discriminador
        for _ in range(DISC_UPDATES):
            self.discriminator_optimizer1.zero_grad()
            prototype = torch.tensor(get_batch_prototypes(real_data['word']), dtype=torch.float32).clone().detach()
            z_repeated = z.unsqueeze(1).repeat(1, 128, 1)  # Repete z para concatenação com protótipos
            gen_input = torch.cat([z_repeated, prototype], dim=-1)
            fake_data = self.generator(gen_input)

            disc_loss_cycle1 = self.discriminator1.disc_loss(fake_data, real_path)
            disc_loss_cycle1.backward()
            # torch.nn.utils.clip_grad_norm_(self.discriminator1.parameters(), max_norm=1.0)
            self.discriminator_optimizer1.step()

        # Treinar o gerador
        self.generator_optimizer.zero_grad()
        gen_input = torch.cat([z_repeated, prototype], dim=-1)
        fake_data = self.generator(gen_input)
        mu_fake, log_var_fake = self.encoder(fake_data)
        z_generated = self.encoder.reparameterize(mu_fake, log_var_fake)
    
        gen_loss1 = self.generator.gen_loss(self.discriminator1, self.encoder, fake_data, real_path, z, z_generated)
        gen_loss1.backward(retain_graph = True)
        # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        # self.generator_optimizer.step()

        # Treinar o segundo discriminador
        for _ in range(DISC_UPDATES):
            self.discriminator_optimizer2.zero_grad()
            mu_real, log_var_real = self.encoder(real_path)
            z_real = self.encoder.reparameterize(mu_real, log_var_real)
            z_repeated = z_real.unsqueeze(1).repeat(1, 128, 1)  # Repete z_real para concatenação com protótipos
            gen_input = torch.cat([z_repeated, prototype], dim=-1)
            fake_data = self.generator(gen_input)
            #printa a diferenca media entre cada ponto da sequencia real e da sequencia falsa

            diferenca = torch.mean(torch.abs(real_path - fake_data))
            # print('real path', real_path)
            print(diferenca)
            disc_loss_cycle2 = self.discriminator2.disc_loss(fake_data, real_path)

            disc_loss_cycle2.backward(retain_graph = True)

            # torch.nn.utils.clip_grad_norm_(self.discriminator2.parameters(), max_norm=1.0)
            self.discriminator_optimizer2.step()

        # Treinar o gerador novamente
        self.generator_optimizer.zero_grad()
        gen_input = torch.cat([z_repeated, prototype], dim=-1)
        fake_data = self.generator(gen_input)

        mu_fake, log_var_fake = self.encoder(fake_data)
        z_generated = self.encoder.reparameterize(mu_fake, log_var_fake)
        gen_loss2 = self.generator.gen_loss(self.discriminator2, self.encoder, fake_data, real_path, z_real, z_generated)
        gen_loss2.backward(retain_graph = True)
        # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        # self.generator_optimizer.step()

        return {'gen_loss1': gen_loss1.item(), 'disc_loss1': disc_loss_cycle1.item(), 'gen_loss2': gen_loss2.item(), 'disc_loss2': disc_loss_cycle2.item()}, fake_data

if __name__ == '__main__':
    # Inicialização do TensorBoard
    writer = SummaryWriter(log_dir='./runs/gesture_generation_experiment')

    # Função de colagem para o DataLoader
    def collate_fn(batch):
        words = [item['word'] for item in batch]
        paths = torch.stack([item['path'] for item in batch])
        return {'word': words, 'path': paths}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print('Iniciando Treinamento...')

    # Inicialização dos modelos
    generator = Generator()
    discriminator1 = Discriminator()
    discriminator2 = Discriminator()
    encoder = VariationalEncoder()

    # Inicialização dos pesos
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    generator.apply(initialize_weights)
    discriminator1.apply(initialize_weights)
    discriminator2.apply(initialize_weights)
    encoder.apply(initialize_weights)

    model = MODEL(generator, discriminator1, discriminator2, encoder)
    batch_count = 0

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        epoch_start_time = time.time()
        for real_data in dataloader:
            iteration_start_time = time.time()

            losses, fake_data = model.train_step(real_data)
            iteration_time = time.time() - iteration_start_time

            # Registro das perdas no TensorBoard
            writer.add_scalar('Generator Loss/First Cycle', losses['gen_loss1'], batch_count)
            writer.add_scalar('Discriminator Loss/First Cycle', losses['disc_loss1'], batch_count)
            writer.add_scalar('Generator Loss/Second Cycle', losses['gen_loss2'], batch_count)
            writer.add_scalar('Discriminator Loss/Second Cycle', losses['disc_loss2'], batch_count)
            writer.add_scalar('Iteration Time', iteration_time, batch_count)

            if batch_count % 10 == 0:
                print(f'Batch {batch_count}: Gen Loss1={losses["gen_loss1"]:.4f}, Disc Loss1={losses["disc_loss1"]:.4f}, '
                      f'Gen Loss2={losses["gen_loss2"]:.4f}, Disc Loss2={losses["disc_loss2"]:.4f}, Time={iteration_time:.4f}s')

                # Seleciona os primeiros 4 gestos do batch para visualização
                real_paths = real_data['path'][:4]
                fake_paths = fake_data[:4]
                words = real_data['word'][:4]

                plot_sequence(real_paths, fake_paths, words, batch_count)

            batch_count += 1

        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1} concluído em {epoch_time:.2f}s')

    # Finaliza o TensorBoard
    writer.close()
    print('Treinamento concluído e logs salvos no TensorBoard.')
