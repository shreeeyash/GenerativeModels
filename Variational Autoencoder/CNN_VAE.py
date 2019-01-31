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
        #=================== For Encoder ================================
        self.conv1 = nn.Conv2d(1, 64, stride=2, padding=(
            15, 15), kernel_size=(4, 4))  # 28*28 to 26*26
        self.conv2 = nn.Conv2d(
            64, 128, stride=2, padding=(15, 15), kernel_size=(4, 4))

        self.l11 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.l12 = nn.Linear(in_features=1024, out_features=20)

        self.l21 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.l22 = nn.Linear(in_features=1024, out_features=20)

        #=================== For Decoder ================================
        self.fc1 = nn.Linear(in_features=20, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=128 * 7 * 7)

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, padding=1, stride=2, kernel_size=4)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=1, padding=1, stride=2, kernel_size=4)

    def Encoder(self, x):
        a1 = F.relu(self.conv1(x))
        a2 = F.relu(self.conv2(a1))
        a2 = a2.view(-1, 128 * 28 * 28)
        mu = F.relu(self.l11(a2))
        mu = self.l12(mu)
        log_var = F.relu(self.l21(a2))
        log_var = self.l22(log_var)

        return mu, log_var

    # ask how do we decide whether its learning var or sigma??
    def Reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # why not std.size()???
        return epsilon.mul(std).add_(mu)

    def Decode(self, z):
        d1 = F.relu(self.fc1(z))
        d2 = F.relu(self.fc2(d1))
        d2 = d2.view(-1, 128, 7, 7)
        d3 = F.relu(self.deconv1(d2))
        d4 = self.deconv2(d3)
        return torch.sigmoid(d4)  # ask why taking sigmoid here???

    def Forward(self, x):
        mu, log_var = self.Encoder(x.view(-1, 1, 28, 28))
        z = self.Reparameterize(mu, log_var)
        return self.Decode(z), mu, log_var


#================= vae is our Model ======================================================
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)


#=============== Define Loss function derived in paper ====================================
def Loss(recons_x, x, mu, log_var):
    #recon_loss = (recons_x-x.view(-1,784)).pow(2).mean()
    #recon_loss = torch.sum((recons_x-x.view(-1,784)).pow(2))
    recon_loss = F.binary_cross_entropy(
        recons_x.view(-1, 784), x.view(-1, 784), reduction='sum')  # why binary & not categorical??
    Dkl = 0.5 * torch.sum(log_var.exp() + mu.pow(2) - 1 - log_var)
    return recon_loss + Dkl

#================ Training ====================================================================


def train(epoch):
    vae.train()   # why vae.train() here???
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recons_batch, mu, log_var = vae.Forward(
            data)  # why not vae.forward()????
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
for ep in range(1, 26):
    train(ep)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = vae.Decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   '/home/shreeyash/Desktop/Deep Learning/Variational Autoencoder/results/binary_crossentropy/sample_' + str(ep) + '.png')


