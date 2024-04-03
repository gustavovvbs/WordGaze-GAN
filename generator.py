import torch
import torch.nn.functional as F
from torch import nn

device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))

class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size, 32, bidirectional=True)
        self.lstm2 = nn.LSTM(32, 32, bidirectional=True)
        self.lstm3 = nn.LSTM(32, 32, bidirectional=True) ####a bidirectional dobra a dimensao do output 
        self.fc1 = nn.Linear(64, 3) ##ai a dimensao daq temq ser 64x3 n 32x3, ou seja, a arquitetura q eles botaram no paper ta errada hehehe
    
    def forward(self, x):
        x, _ = self.lstm1(x) #[128, 64] #lstm da output de um tuple ai temq pegar em forma de tuple pq o x vai ser o hidden state q vai ser o output
        x, _ = self.lstm2(x) #[128, 64]
        x, _ = self.lstm3(x) #[128, 64]


        output = self.fc1(x)
        output = torch.tanh(output)

        return output #[128, 3]


if __name__ == '__main__':
    model = Generator(input_size=35) #cria a instancia do modelo
    print('dimension of the output', model(torch.randn(128,35)).shape) #chama forward e pega o shape

