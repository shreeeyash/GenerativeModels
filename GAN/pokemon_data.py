import torch
import torchvision


class load_data():
    def __init__(self, Transform=None, batch_size=32):
        self.dataset = torchvision.datasets.ImageFolder(
            root="/home/shreeyash/Desktop/Deep Learning/Generative Models/GAN/data",
            transform=Transform)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True)

    def get_data(self):
        return {"dataset": self.dataset, "dataloader": self.dataloader}


"""
x = load_data()
print(x.get_data())
"""
