import torch 
from torch import nn 
import torchvision 

from dataloader import * 
from utils import * 
from model import * 

import matplotlib.pyplot as plt 
import os 


keyboard_layout = {
    'q': (-0.9, 0.9), 'w': (-0.7, 0.9), 'e': (-0.5, 0.9), 'r': (-0.3, 0.9), 't': (-0.1, 0.9),
    'y': (0.1, 0.9), 'u': (0.3, 0.9), 'i': (0.5, 0.9), 'o': (0.7, 0.9), 'p': (0.9, 0.9),
    'a': (-0.85, 0.7), 's': (-0.65, 0.7), 'd': (-0.45, 0.7), 'f': (-0.25, 0.7), 'g': (-0.05, 0.7),
    'h': (0.15, 0.7), 'j': (0.35, 0.7), 'k': (0.55, 0.7), 'l': (0.75, 0.7),
    'z': (-0.8, 0.5), 'x': (-0.6, 0.5), 'c': (-0.4, 0.5), 'v': (-0.2, 0.5), 'b': (0.0, 0.5),
    'n': (0.2, 0.5), 'm': (0.4, 0.5)
}

key_size = 0.08  # Adjust this value for the visual size of each key

def plot_keyboard(ax):
    for key, (x, y) in keyboard_layout.items():
        rect = plt.Rectangle((x - key_size / 2, y - key_size / 2), key_size, key_size, edgecolor='black', facecolor='lightgray')
        ax.add_patch(rect)
        ax.text(x, y, key, ha='center', va='center', fontsize=10)

""" loss de waserstein 
    parametros:
    - real: dado real
    - fake: dado falso
"""

def wasserstein_loss(fake, real):
    dtype = type(fake)

    loss = torch.mean(fake) - torch.mean(real)

    return loss 

def l1_loss(fake, real):
    return torch.mean(torch.abs(fake - real))


class Solver():
    def __init__(self, root = 'gestures_data.json', result_dir = 'C:/Users/gugu1/Documents/GitHub/swipetest/experiments/processed_gestures.json', weight_dir = 'weight', load_weight = False, batch_size = 256, test_size = 20, num_epoch = 100, save_every = 1000, lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999, lambda_kl = 0.05, lambda_rec = 5, z_dim = 32, lambda_lat = 0.5):

        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.dloader, dlen = dataloader(root, batch_size = batch_size, test_size = test_size, mode = 'train')

        self.t_dloader, _ = dataloader(root, batch_size = test_size, test_size = test_size, mode = 'test')

        #definindo os modelos 
        # D_cVAE: discriminador do cVAE (ENCODA O DADO REAL COMO CONDICIONAL)
        # D_cLR: discriminador do cLR (RECUPERA O ESPACO LATENTE AMOSTRADO DE UMA NORMAL)

        self.D_cVAE = Discriminator().type(self.dtype).to('cuda')
        self.D_cLR = Discriminator().type(self.dtype).to('cuda')
        self.G = Generator(z_dim).type(self.dtype).to('cuda')
        self.E = Encoder().type(self.dtype).to('cuda')

        self.optim_D_cVAE = torch.optim.Adam(self.D_cVAE.parameters(), lr = lr, betas = (beta_1, beta_2))

        self.optim_D_cLR = torch.optim.Adam(self.D_cLR.parameters(), lr = lr, betas = (beta_1, beta_2))

        self.optim_G = torch.optim.Adam(self.G.parameters(), lr = lr, betas = (beta_1, beta_2))

        self.optim_E = torch.optim.Adam(self.E.parameters(), lr = lr, betas = (beta_1, beta_2))

        self.fixed_z = var(torch.randn(256, z_dim), requires_grad = False)

        self.z_dim = z_dim 
        self.lambda_kl = lambda_kl 
        self.lambda_lat = lambda_lat 
        self.lambda_rec = lambda_rec

        self.result_dir = result_dir
        self.weight_dir = weight_dir
        self.load_weight = load_weight 
        self.test_size = test_size
        self.save_every = save_every
        self.num_epoch = num_epoch

    def show_model(self):
        print('========================== Discriminador do cVAE ==========================')
        print(self.D_cVAE)
        print('========================== Discriminador do cLR ==========================')
        print(self.D_cLR)
        print('========================== Gerador ==========================')
        print(self.G)
        print('========================== Encoder ==========================')
        print(self.E)

    def set_train_phase(self):
        self.D_cLR.train()
        self.D_cVAE.train()
        self.G.train()
        self.E.train()

    def load_pretrained(self):
        self.D_cVAE.load_state_dict(torch.load(os.path.join(self.weight_dir, 'D_cVAE.pkl')))
        self.D_cLR.load_state_dict(torch.load(os.path.join(self.weight_dir, 'D_cLR.pkl')))
        self.G.load_state_dict(torch.load(os.path.join(self.weight_dir, 'G.pkl')))
        self.E.load_state_dict(torch.load(os.path.join(self.weight_dir, 'E.pkl')))
        
        log_file = open('log.txt', 'r')
        line = log_file.readline()
        self.start_epoch = int(line)

    ''' save weight ''' 

    def save_weight(self, epoch = None):
        if epoch is None:
            d_cVAE_name = 'D_cVAE.pkl'
            d_cLR_name = 'D_cLR.pkl'
            g_name = 'G.pkl'
            e_name = 'E.pkl'
        else:
            d_cVAE_name = '{epochs}-{name}'.format(epochs=str(epoch), name='D_cVAE.pkl')
            d_cLR_name = '{epochs}-{name}'.format(epochs=str(epoch), name='D_cLR.pkl')
            g_name = '{epochs}-{name}'.format(epochs=str(epoch), name='G.pkl')
            e_name = '{epochs}-{name}'.format(epochs=str(epoch), name='E.pkl')
            
        torch.save(self.D_cVAE.state_dict(), os.path.join(self.weight_dir, d_cVAE_name))
        torch.save(self.D_cVAE.state_dict(), os.path.join(self.weight_dir, d_cLR_name))
        torch.save(self.G.state_dict(), os.path.join(self.weight_dir, g_name))
        torch.save(self.E.state_dict(), os.path.join(self.weight_dir, e_name))

    ''' zera o gradiente de todos os modelos ''' 

    def all_zero_grad(self):
        self.optim_D_cLR.zero_grad()
        self.optim_D_cVAE.zero_grad()
        self.optim_G.zero_grad()
        self.optim_E.zero_grad()

    def train(self):
        # if self.load_pretrained:
        #     self.load_pretrained()
        
        self.set_train_phase()
        self.show_model()

        for epoch in range(0, self.num_epoch):
            for iters, real_data in enumerate(self.dloader):
                prototype = var(get_batch_prototypes(real_data[1]))
                real_path = var(real_data[0]).type(self.dtype).to('cuda')
                # ========================= Discriminator do cVAE =========================
                self.all_zero_grad()
                mu, log_var = self.E(real_path)
                random_z = var(torch.randn(real_path.size(0), self.z_dim)).type(self.dtype).to('cuda')
                std = torch.exp(log_var / 2)
                z = mu + std * random_z
                fake_gesture_cVAE = self.G(prototype, z)

                real_pair_cVAE = real_path
                fake_pair_cVAE = fake_gesture_cVAE

                real_cVAE, _ = self.D_cVAE(real_pair_cVAE)
                fake_cVAE, _ = self.D_cVAE(fake_pair_cVAE)

                loss_D_cVAE = wasserstein_loss(fake_cVAE, real_cVAE)

                # ========================= Discriminator do cLR =========================
                fake_gesture_cLR = self.G(prototype, random_z)
                real_pair_cLR = real_path
                fake_pair_cLR = fake_gesture_cLR

                D_cLR_real, _ = self.D_cLR(real_pair_cLR)
                D_cLR_fake, _ = self.D_cLR(fake_pair_cLR)

                loss_D_cLR = wasserstein_loss(D_cLR_fake, D_cLR_real)
                D_loss = loss_D_cVAE + loss_D_cLR

                self.all_zero_grad()
                D_loss.backward()
                self.optim_D_cVAE.step()
                self.optim_D_cLR.step()

                # ========================= Gerador e encoder =========================
                real_path = var(real_data[0]).type(self.dtype).to('cuda')
                mu, log_var = self.E(real_path)
                std = torch.exp(log_var / 2)
                random_z = var(torch.randn(real_path.size(0), self.z_dim)).type(self.dtype).to('cuda')
                z = mu + std * random_z

                D_cLR_real, real_activations = self.D_cLR(real_path)

                fake_gesture_cVAE = self.G(prototype, z)
                fake_pair_cVAE = fake_gesture_cVAE
                D_cLR_fake, _ = self.D_cLR(fake_pair_cVAE)
                G_GAN_LOSS_CVAE = -torch.mean(D_cLR_fake)

                #gerar pra clr agr 
                random_z = var(torch.randn(real_path.size(0), self.z_dim)).type(self.dtype).to('cuda')
                fake_gesture_cLR = self.G(prototype, random_z)
                fake_pair_cLR = fake_gesture_cLR
                D_cLR_fake, activations_fake = self.D_cLR(fake_pair_cLR)
                G_GAN_LOSS_CLR = -torch.mean(D_cLR_fake)

                G_GAN_LOSS = G_GAN_LOSS_CVAE + G_GAN_LOSS_CLR

                #####Botar a kld pro encoder 
                KL_DIV = self.lambda_kl * torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))

                img_rec_loss = self.lambda_rec * l1_loss(fake_gesture_cVAE, real_path)

                z_feat_loss = 0
                for real_act, fake_act in zip(real_activations, activations_fake):
                    N_i = real_act.numel()
                    z_feat_loss += l1_loss(real_act, fake_act) / N_i
                    
                EG_LOSS = G_GAN_LOSS + KL_DIV + img_rec_loss + z_feat_loss

                self.all_zero_grad()
                EG_LOSS.backward()
                self.optim_G.step()
                self.optim_E.step()

                ''' treinar so o gerador na recursiva latente ''' 
           
                fake_img_cLR = self.G(prototype, random_z)
                mu, log_var = self.E(fake_img_cLR)
                z_rec_loss = self.lambda_lat * l1_loss(mu, random_z)
               




              
                

                self.all_zero_grad()
                z_rec_loss.backward()
                self.optim_G.step()

            # Print error, save intermediate result image and weight
                if iters % self.save_every == 0:
                    print('[Epoch : %d / Iters : %d] => D_loss : %f / G_GAN_loss : %f / KL_div : %f / img_recon_loss : %f / z_recon_loss : %f / z_feat_loss : %f'
                            %(epoch, iters, D_loss.item(), G_GAN_LOSS.item(), KL_DIV.item(), img_rec_loss.item(), z_rec_loss.item(), z_feat_loss.item()))

                    # Save intermediate result image
                    if os.path.exists(self.result_dir) is False:
                        os.makedirs(self.result_dir)


                    generated_gesture = self.G(prototype, self.fixed_z)

                    generated_gesture = generated_gesture.cpu().detach().numpy()
                    generated_X, generated_Y = generated_gesture[:, :, 0], generated_gesture[:, :, 1]

                    for i in range(self.test_size):
                        fig, ax = plt.subplots()
                        plot_keyboard(ax)
                        ax.plot(generated_X[i], generated_Y[i], marker='o',label='generated')
                        ax.plot(real_data[0][i][:, 0], real_data[0][i][:, 1], marker='o', label='real')

                        # Set limits to [-1, 1] to match the normalized coordinates
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(-1, 1)

                        # Add title and legend
                        plt.title('Generated gesture, word: {word}'.format(word=real_data[1][i]))
                        plt.legend()

                        # Save the plot
                        plt.savefig(os.path.join(self.result_dir, 'Generated_gesture_epoch_{epoch}_iter_{iters}.png'.format(epoch=epoch, iters=iters)))
                        plt.close()

                    # # Save intermediate weight
                    # if os.path.exists(self.weight_dir) is False:
                    #     os.makedirs(self.weight_dir)
                    
                    







