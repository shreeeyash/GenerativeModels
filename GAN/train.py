import torch
import torchvision
import torchsummary
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from pokemon_data import load_data
from model import Descriminator, Generator

# load data
# subtract 0.5 and then divide by 0.5 yielding an image with mean zero and values in range [-1, 1]
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
X = load_data(Transform=transform, batch_size=128)
X = X.get_data()
dataset = X["dataset"]
dataloader = X["dataloader"]


# load model for generator and descriminator
G = Generator().cuda()
D = Descriminator().cuda()
print(torchsummary.summary(G, (100, 1, 1)))
print(torchsummary.summary(D, (3, 64, 64)))


# weight initialization as given in DCGAN paper
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, mean=0, std=0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)


G.apply(init_weights)
D.apply(init_weights)


# Training setup
epochs = 111
z_const = Variable(torch.randn((16, 100, 1, 1)).cuda())
optim_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_D = torch.optim.SGD(D.parameters(), lr=0.0002, momentum=0.9)
BCE = torch.nn.BCELoss()

for epoch in range(epochs):
    i = 0
    # logs for storing data during training
    dir_path = "/home/shreeyash/Desktop/Deep Learning/Generative Models/GAN/log_dir/new data set/epoch_{0}".format(
        epoch + 1)
    print(dir_path)
    writer = SummaryWriter(dir_path)
    for x, _ in dataloader:
        i = i + 1
        minibatch_size = x.size()[0]
        y_real = Variable(torch.ones(minibatch_size).cuda())
        y_fake = Variable(torch.zeros(minibatch_size).cuda())
        x = Variable(x.cuda())

        # discriminator
        D.zero_grad()
        # Real loss
        real_loss = BCE(D(x).squeeze(), y_real)
        # Fake loss
        z = Variable(torch.randn((minibatch_size, 100)
                                 ).view(-1, 100, 1, 1).cuda())
        fake_loss = BCE(D(G(z)).squeeze(), y_fake)
        # Total discriminator loss
        D_loss = fake_loss + real_loss
        D_loss.backward()
        optim_D.step()

        # Generator
        G.zero_grad()
        z = Variable(torch.randn((minibatch_size, 100)
                                 ).view(-1, 100, 1, 1).cuda())
        G_loss = BCE(D(G(z)).squeeze(), y_real)
        G_loss.backward()
        optim_G.step()

        # logs
        writer.add_scalar("DiscriminatorLoss", D_loss, i)
        writer.add_scalar("GeneratorLoss", G_loss, i)

        # information printing
        print("iteration:", i, " epoch:", epoch, " D_loss:",
              D_loss.item(), " G_loss:", G_loss.item())
        # Generated images
        with torch.no_grad():
            generated_img = G(z_const).detach()
        writer.add_image("Generated Image from const noise", torchvision.utils.make_grid(
            generated_img, nrow=4, normalize=True), i)
        # tensorboard --logdir='./logs' --port=6006

writer.close()
