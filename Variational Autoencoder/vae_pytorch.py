import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import cv2

device = torch.device("cuda")

#======= Loading dataset of 60000 mnist images in train_loader====================================
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                                          transform=transforms.ToTensor()), batch_size=50, shuffle=True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False,
                                                         transform=transforms.ToTensor()), batch_size=50, shuffle=True)


#============= Defining the model in class ========================================================

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.l1 = nn.Linear(784, 400)
        self.l21 = nn.Linear(400, 20)
        self.l22 = nn.Linear(400, 20)
        self.l3 = nn.Linear(20, 400)
        self.l4 = nn.Linear(400, 784)

    def Encoder(self, x):
        a1 = F.relu(self.l1(x))
        return self.l21(a1), self.l22(a1)

    
    def Reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  
        return epsilon.mul(std).add_(mu)

    def Decode(self, z):
        a3 = F.relu(self.l3(z))
        return torch.sigmoid(self.l4(a3))  

    def Forward(self, x):
        mu, log_var = self.Encoder(x.view(-1, 784))
        z = self.Reparameterize(mu, log_var)
        return self.Decode(z), mu, log_var


#================= vae is our Model ======================================================
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)


#=============== Define Loss function derived in paper ====================================
def Loss(recons_x, x, mu, log_var):
    #recon_loss = (recons_x-x.view(-1,784)).pow(2).mean() ~~compare KLD and P(x|z) losses
    #recon_loss = torch.sum((recons_x-x.view(-1,784)).pow(2))
    recon_loss = F.binary_cross_entropy(recons_x, x.view(-1, 784), reduction='sum') 
    Dkl = 0.5 * torch.sum(log_var.exp() + mu.pow(2) - 1 - log_var)
    return recon_loss + Dkl

#================ Training ====================================================================


def train(epoch):
    vae.train()   
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recons_batch, mu, log_var = vae.Forward(data) 
        loss = Loss(recons_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


#================ Inference ==============================================
for ep in range(1,26):
	train(ep)
	with torch.no_grad():
		sample = torch.randn(64,20).to(device)
		sample = vae.Decode(sample).cpu()
		save_image(sample.view(64,1,28,28),'/home/shreeyash/Desktop/Deep Learning/Variational Autoencoder/results/binary_crossentropy/sample_'+str(ep)+'.png')
