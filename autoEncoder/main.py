# original image - encode - decode - reconstructed image
# for encoding and decoding we use --> Neural net
# nn.conTranspose2d

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

# transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5), (0.5))
# ])
# image value will be between -1 to +1. so we use nn.Tanh

mnist_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
data_loader = torch.utils.data.DataLoader(dataset = mnist_data, batch_size = 64, shuffle = True)

dataiter = iter(data_loader)
images, labels = next(dataiter)
print(torch.min(images), torch.max(images))# values is between 0 and 1. It can change we we give another value for transforms.ToTensor()

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # in the beginning our images size is N(batch size), 784
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),# reduce size by (N, 784 -> N, 128)
            nn.ReLU(),# activation method
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3) # -> N, 3
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),  # N, 3 --> N, 128
            nn.Sigmoid()# Since the image value is between 0 and 1, we use this activation method to enxure this

        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-5)

# Training Loop
num_epochs = 10
outputs = []
for epoch in range(num_epochs):
    for(img, _) in data_loader:
        img = img.reshape(-1, 28*28)
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch, img, recon))

# plot images to see how the reconstructed image looks like
for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, items in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        item = items.reshape(-1, 28, 28)
        plt.imshow(item[0])
    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1)
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])

