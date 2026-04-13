"""
Loss formulation
----------------
We use the Least-Squares GAN (LSGAN, Mao et al. 2017) objective:

    Discriminator:
        L_D = 0.5 * E[( D(real) - 1 )²]   (real should score 1)
            + 0.5 * E[( D(fake) - 0 )²]   (fake should score 0)

    Generator:
        L_G_adv = 0.5 * E[( D(fake) - 1 )²]  (fool D → score fake as 1)

LSGAN is more training-stable than the original minimax GAN because the
gradients do not vanish when the discriminator is confident.
"""

import torch
import torch.nn as nn


class LSGANDiscriminatorLoss(nn.Module):
    """
    LSGAN discriminator objective.
    L_D = 0.5 * MSE(D(real), 1) + 0.5 * MSE(D(fake), 0)

    Args:
        loss_weight (float): Scalar applied to the total discriminator loss. Typically 0.5-1.0.
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.mse = nn.MSELoss()

    def forward(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_real: Discriminator output on real images  (B, 1, H', W').
            pred_fake: Discriminator output on fake (generated) images.
                       Must be detached from the generator graph before being passed here.
        Returns:
            Scalar discriminator loss.
        """
        real_loss = self.mse(pred_real, torch.ones_like(pred_real))
        fake_loss = self.mse(pred_fake, torch.zeros_like(pred_fake))
        return self.loss_weight * 0.5 * (real_loss + fake_loss)


class LSGANGeneratorLoss(nn.Module):
    """
    LSGAN generator adversarial objective.
    L_G_adv = 0.5 * MSE(D(fake), 1)

    Args:
        loss_weight (float): Weight applied to this adversarial term before adding it to the full generator loss.
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.mse = nn.MSELoss()

    def forward(self, pred_fake: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_fake: Discriminator output on fake images (B, 1, H', W'). Should NOT be detached - gradients flow back to G.
        Returns:
            Scalar generator adversarial loss.
        """
        return self.loss_weight * 0.5 * self.mse(pred_fake, torch.ones_like(pred_fake))


# Convenience factory
def build_gan_losses(d_weight_rgb: float = 1.0, d_weight_hvi: float = 1.0, g_weight_rgb: float = 1.0, g_weight_hvi: float = 1.0,):
    
    # Instantiate all four GAN loss objects and return them as a dict.
    return {
        'D_rgb': LSGANDiscriminatorLoss(loss_weight=d_weight_rgb).cuda(),
        'D_hvi': LSGANDiscriminatorLoss(loss_weight=d_weight_hvi).cuda(),
        'G_rgb': LSGANGeneratorLoss(loss_weight=g_weight_rgb).cuda(),
        'G_hvi': LSGANGeneratorLoss(loss_weight=g_weight_hvi).cuda(),
    }
