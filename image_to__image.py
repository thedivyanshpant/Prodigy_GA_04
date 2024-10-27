

import os
import glob
import itertools
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

# U-Net Generator
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(3, 64, normalize=False),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            self.conv_block(512, 512),
            self.conv_block(512, 512),
            self.conv_block(512, 512),
            self.conv_block(512, 512, normalize=False)
        )
        self.decoder = nn.Sequential(
            self.deconv_block(512, 512, dropout=0.5),
            self.deconv_block(1024, 512, dropout=0.5),
            self.deconv_block(1024, 512, dropout=0.5),
            self.deconv_block(1024, 512),
            self.deconv_block(1024, 256),
            self.deconv_block(512, 128),
            self.deconv_block(256, 64)
        )
        self.final = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def conv_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def deconv_block(self, in_channels, out_channels, dropout=0.0):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        enc_results = [x]
        for layer in self.encoder:
            enc_results.append(layer(enc_results[-1]))

        dec_input = enc_results[-1]
        for i, layer in enumerate(self.decoder):
            dec_input = torch.cat([layer(dec_input), enc_results[-(i + 2)]], 1)

        return self.tanh(self.final(dec_input))

# PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            self.conv_block(6, 64, normalize=False),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def conv_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Loss functions
adversarial_loss = nn.MSELoss()
pixelwise_loss = nn.L1Loss()

# Initialize models
generator = UNetGenerator().cuda()
discriminator = PatchGANDiscriminator().cuda()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = self.transform(img.crop((0, 0, w / 2, h)))
        img_B = self.transform(img.crop((w / 2, 0, w, h)))

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)

# DataLoader
transform = [
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

dataloader = DataLoader(ImageDataset('dataset', transforms_=transform), batch_size=1, shuffle=True)
val_dataloader = DataLoader(ImageDataset('dataset', transforms_=transform, mode='val'), batch_size=1, shuffle=True)

# Training
n_epochs = 200
sample_interval = 100

for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].cuda()
        real_B = batch['B'].cuda()

        # Adversarial ground truths
        valid = torch.ones((real_A.size(0), 1, 30, 30), requires_grad=False).cuda()
        fake = torch.zeros((real_A.size(0), 1, 30, 30), requires_grad=False).cuda()

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(torch.cat((real_A, fake_B), 1))
        loss_GAN = adversarial_loss(pred_fake, valid)

        # Pixel-wise loss
        loss_pixel = pixelwise_loss(fake_B, real_B)

        # Total loss
        loss_G = loss_GAN + 100 * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(torch.cat((real_A, real_B), 1))
        loss_real = adversarial_loss(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(torch.cat((real_A, fake_B.detach()), 1))
        loss_fake = adversarial_loss(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # Log progress
        print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

        # Save sample images
        if i % sample_interval == 0:
            save_image(fake_B.data, f"images/{epoch}_{i}.png", normalize=True)

# Testing
def sample_images(epoch):
    real_A = next(iter(val_dataloader))['A'].cuda()
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data), -2)
    save_image(img_sample, f"images/{epoch}.png", nrow=5, normalize=True)

sample_images(n_epochs)
