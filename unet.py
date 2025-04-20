import torch
import torch.nn as nn
import torch.linalg as la
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np

class UNet(nn.Module):
    """ 
    A simple U-Net architecture for image segmentation.
    Based on the U-Net architecture from the original paper:
    Olaf Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, in_channels=3, n_classes=19):
        
        super(UNet, self).__init__()

        self.image_size = 518
        self.patch_size = 14
        self.embed_dim = 384

        self.dino_encoder = DinoEncoder(self.image_size, self.patch_size, self.embed_dim)
        self.encoder = HierarchicalEncoder(in_channels)
        self.seg_decoder = SegmentationDecoder(n_classes)

    def forward(self, x):
        x4, x3, x2, x1 = self.encoder(x)
        x5 = self.dino_encoder(x)

        features = [x5, x4, x3, x2, x1] 

        # Segmentation output
        logits, _ = self.seg_decoder(features)

        means=np.load('gmm_means.npy')
        covs=np.load('gmm_cov.npy')
        weights=np.load('gmm_weights.npy')
        feature_map = x4.mean(axis=(2, 3)).detach().cpu().numpy()
        gmm = GaussianMixture(n_components=len(means), covariance_type='full')
        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))
        gmm.weights_ = weights
        gmm.means_ = means
        gmm.covariances_ = covs

        score=gmm.score_samples(feature_map)
        in_distribution=True
        if score[0] < THRESHOLD:
            in_distribution=False

        return logits, in_distribution

class HierarchicalEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(HierarchicalEncoder, self).__init__()

        self.inc = (TripleConvolution(in_channels, 64, padding=0))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 384))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x4, x3, x2, x1
    
class DinoEncoder(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim):
        super(DinoEncoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # Set the model to evaluation mode
        self.dino.eval()
        
        # Freeze all parameters of the DinoV2 encoder
        for param in self.dino.parameters():
            param.requires_grad = False

        # Unfreeze the last two blocks of the DinoV2 encoder for fine-tuning
        # for block in self.dino.blocks[-2:]:
        #     for param in block.parameters():
        #         param.requires_grad = True

        self.triple_conv = TripleConvolution(384, 384, padding=0)

    def forward(self, x):
        x = self.dino.forward_features(x)['x_norm_patchtokens']

        path_h, path_w = int(self.image_size / self.patch_size), int(self.image_size / self.patch_size)
        x = x.permute(0, 2, 1).reshape(-1, self.embed_dim, path_h, path_w)
        x = self.triple_conv(x)
        return x
    
class SegmentationDecoder(nn.Module):
    def __init__(self, n_classes=1):
        super(SegmentationDecoder, self).__init__()

        self.up1 = (Up(768, 256, target_size=(64, 64)))
        self.up2 = (Up(512, 128))
        self.up3 = (Up(256, 64))
        self.up4 = (Up(128, 64))
        self.up5 = nn.Upsample(size=(518, 518), mode='bilinear', align_corners=True)
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x5, x4, x3, x2, x1 = x
        x_base = self.up1(x5, x4)
        x = self.up2(x_base, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        logits = self.outc(x)

        return logits, x_base

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class TripleConvolution(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, target_size=None):
        super().__init__()
        if target_size:
            self.up = nn.Upsample(size=target_size, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        

def compute_input_gradients(model, x, pseudo_labels):
    """Compute gradients of the loss with respect to the input x for given pseudo_labels."""
    model.eval()
    # Ensure x requires gradients and is detached from previous graphs
    x = x.clone().detach().requires_grad_(True)
    logits, _ = model(x)  # Fresh forward pass
    loss = F.cross_entropy(logits, pseudo_labels, reduction='mean')
    model.zero_grad()
    loss.backward()
    return x.grad.clone()

def gsa_ood_detection(model, image, num_perturbations=3, noise_std=0.01):
    """Perform Gradient Sensitivity Analysis for OOD detection."""
    model.eval()
    
    # Step 1: Compute pseudo-labels without gradients
    with torch.no_grad():
        logits, _ = model(image)
        pseudo_labels = torch.argmax(logits, dim=1)
    
    # Step 2: Compute baseline gradients
    baseline_grads = compute_input_gradients(model, image, pseudo_labels)
    
    # Step 3: Compute perturbed gradients
    sensitivity_scores = []
    for _ in range(num_perturbations):
        # Recompute logits with a fresh graph for perturbation
        image_pert = image.clone().detach().requires_grad_(True)
        logits, _ = model(image_pert)
        # Perturb logits
        perturbed_logits = logits + torch.randn_like(logits) * noise_std
        # Compute gradients using perturbed logits
        loss = F.cross_entropy(perturbed_logits, pseudo_labels, reduction='mean')
        model.zero_grad()
        loss.backward()
        perturbed_grads = image_pert.grad.clone()
        # Compute gradient difference
        grad_diff = (perturbed_grads - baseline_grads).pow(2).sum(dim=1).sqrt()
        sensitivity_scores.append(grad_diff)
    
    # Step 4: Compute sensitivity and OOD maps
    sensitivity_map = torch.stack(sensitivity_scores, dim=0).mean(dim=0)
    ood_map = (sensitivity_map > sensitivity_map.mean() + 2 * sensitivity_map.std()).float()
    
    return ood_map, sensitivity_map

def compute_gradient_norm_ood(model, input_tensor):
    model.eval()
    input_tensor.requires_grad_(True)
    
    # Forward pass
    output, _ = model(input_tensor)
    pseudo_labels = torch.argmax(output, dim=1)

    # Compute loss
    loss = F.cross_entropy(output, pseudo_labels, reduction='mean')
    
    # Compute gradients w.r.t. features
    model.zero_grad()
    loss.backward()
    
    # Get gradients of the feature layer
    grad = input_tensor.grad
    
    # Compute L2 norm of gradients
    grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3))  # Adjust dims based on your input
    
    return grad_norm