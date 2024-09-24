import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

class Generator(nn.Module):
    def __init__(self, width=32) -> None:
        super().__init__()
        self.in_layer = nn.Linear(100, 14 * 4 * 4)  # Project latent vector to 14 x 4 x 4
        self.net = nn.Sequential(
            nn.Conv2d(14, width * 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(width * 16),
            nn.ConvTranspose2d(width * 16, width * 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(width * 16, width * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(width * 8),
            nn.ConvTranspose2d(width * 8, width * 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(width * 8, width * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(width * 4),
            nn.ConvTranspose2d(width * 4, width * 4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(width * 4, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(width * 2),
            nn.ConvTranspose2d(width * 2, width * 2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(width * 2, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.in_layer(z)           # Linear layer to expand the latent vector
        x = x.view(-1, 14, 4, 4)       # Reshape to 4x4 feature map with 14 channels
        x = self.net(x)                # Pass through convolutional layers
        return x                       # Output the generated image

class Discriminator(nn.Module):
    def __init__(self, width=32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(width * 2),
            nn.Conv2d(width * 2, width * 2, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(width * 4),
            nn.Conv2d(width * 4, width * 4, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(width * 4, width * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(width * 8),
            nn.Conv2d(width * 8, width * 8, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(width * 8, width * 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(width * 16),
            nn.Conv2d(width * 16, width * 16, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(width * 16, width, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(width),
            nn.LeakyReLU(),
            nn.Conv2d(width, 14, kernel_size=3, padding=1)
        )
        self.proj = nn.Linear(14*4*4, 1)  # Linear projection to a scalar (real/fake decision)
    
    def forward(self, img):
        emb = self.net(img)               # Pass the image through the convolutional layers
        emb = emb.view(emb.size(0), -1)   # Flatten the output (batch_size, 14 * 4 * 4)
        out = F.sigmoid(self.proj(emb))              # Pass through the projection layer
        return out                        # Output a single scalar value (real/fake prediction)


# Unit test to verify the generator and discriminator
class TestGeneratorDiscriminator(unittest.TestCase):
    def test_generator_output_shape(self):
        latent_dim = 100
        batch_size = 8
        width = 32
        gen = Generator(width)

        # Generate random latent vectors
        z = torch.randn(batch_size, latent_dim)

        # Generate images
        output = gen(z)

        # Check that output shape is (batch_size, 3, 64, 64)
        self.assertEqual(output.shape, (batch_size, 3, 64, 64))

    def test_discriminator_output_shape(self):
        batch_size = 8
        width = 32
        disc = Discriminator(width)

        # Create random images of shape (batch_size, 3, 64, 64)
        images = torch.randn(batch_size, 3, 64, 64)

        # Pass images through the discriminator
        output = disc(images)

        # Check that the output shape is (batch_size, 1)
        self.assertEqual(output.shape, (batch_size, 1))


if __name__ == '__main__':
    # Run the unit tests
    unittest.main(argv=[''], exit=False)
