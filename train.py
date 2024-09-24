import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import os
import random
import datetime
import matplotlib.pyplot as plt
from model import Generator, Discriminator

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CustomDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label as it's not used

def get_data_loader(path, batch_size, num_samples=None, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # GANs typically use 64x64 or higher resolution
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization for GANs
    ])
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_extensions)]
    
    if len(image_files) == 0:
        raise ValueError("No valid image files found in the specified directory.")

    if num_samples is None or num_samples > len(image_files):
        num_samples = len(image_files)
    elif num_samples <= 0:
        raise ValueError("num_samples should be a positive integer.")

    print("data length: ", len(image_files))

    if shuffle:
        indices = random.sample(range(len(image_files)), num_samples)
    else:
        indices = list(range(num_samples))

    subset_dataset = CustomDataset([image_files[i] for i in indices], transform=transform)
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader

def show_generated_image(generator, device, latent_dim):
    generator.eval()  # Set generator to evaluation mode
    with torch.no_grad():
        # Generate a random latent vector
        z = torch.randn(1, latent_dim).to(device)
        generated_img = generator(z).squeeze(0).cpu()

    # Unnormalize the image (reverse the normalization done during preprocessing)
    generated_img = generated_img * 0.5 + 0.5  # Unnormalize

    # Convert to numpy format
    generated_img = generated_img.permute(1, 2, 0).numpy()

    # Display the generated image using matplotlib
    plt.imshow(generated_img)
    plt.axis('off')  # Hide axes
    plt.show()

def training_loop(n_epochs, generator, discriminator, optimizer_G, optimizer_D, loss_fn, device, data_loader, latent_dim=100, checkpoint_path=None, start_epoch = 0):
    for epoch in range(start_epoch, n_epochs):
        generator.train()
        discriminator.train()
        loss_G_total = 0.0
        loss_D_total = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit=' batch')

        for batch_idx, (real_imgs, _) in enumerate(progress_bar):
            real_imgs = real_imgs.to(device)

            # ======= Train Discriminator =======
            optimizer_D.zero_grad()

            # Create real and fake labels (real=1, fake=0)
            real_labels = torch.ones((real_imgs.size(0), 1)).to(device)
            fake_labels = torch.zeros((real_imgs.size(0), 1)).to(device)

            # Generate fake images
            z = torch.randn(real_imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(z)

            # Discriminator loss on real and fake images
            real_loss = loss_fn(discriminator(real_imgs), real_labels)
            fake_loss = loss_fn(discriminator(fake_imgs.detach()), fake_labels)
            loss_D = (real_loss + fake_loss) / 2

            loss_D.backward()
            optimizer_D.step()

            # ======= Train Generator =======
            optimizer_G.zero_grad()

            # Generator loss - want discriminator to classify fake images as real
            fake_labels_for_G = torch.ones((real_imgs.size(0), 1)).to(device)  # Generator wants to trick the discriminator
            loss_G = loss_fn(discriminator(fake_imgs), fake_labels_for_G)

            loss_G.backward()
            optimizer_G.step()

            loss_G_total += loss_G.item()
            loss_D_total += loss_D.item()

            progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

        print('{} Epoch {}, Generator loss: {}, Discriminator loss: {}'.format(
            datetime.datetime.now(), epoch, loss_G_total / len(data_loader), loss_D_total / len(data_loader)
        ))

        # Show generated image every 5 epochs
        if epoch % 10 == 0:
            print(f"Showing generated image for epoch {epoch}")
            show_generated_image(generator, device, latent_dim)

        # Save model checkpoints every 20 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, f'gan_checkpoint_epoch_{epoch}.pth')

if __name__ == '__main__':
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    latent_dim = 100
    batch_size = 64
    n_epochs = 1000
    lr = 0.0002

    # Initialize Generator and Discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    print("param count gen: ", count_parameters(generator))
    print("param count disc: ", count_parameters(discriminator))

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    loss_fn = nn.BCELoss()

    # DataLoader
    path = '/Users/ayanfe/Documents/Datasets/Waifus/Train'
    data_loader = get_data_loader(path, batch_size=batch_size)

    # Resume training if checkpoint exists
    checkpoint_path = 'gan_checkpoint_epoch_40.pth'

    # Load checkpoint if available
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")


    show_generated_image(generator, device, latent_dim)

    # Training loop
    training_loop(
        n_epochs=n_epochs,
        generator=generator,
        discriminator=discriminator,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        loss_fn=loss_fn,
        device=device,
        data_loader=data_loader,
        latent_dim=latent_dim,
        checkpoint_path=checkpoint_path,
        start_epoch= start_epoch
    )
