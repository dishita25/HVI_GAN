"""
train_gan.py
============
HVI-GAN training script.

Architecture recap
------------------
  Generator (G)        : HVIGANGenerator  – same dual-channel (HV + I)
                          U-Net with LCA cross-attention as CIDNet.
  Discriminator D_rgb  : 70×70 PatchGAN operating on RGB images.
  Discriminator D_hvi  : 70×70 PatchGAN operating on HVI images.

Training objectives
-------------------
  Generator loss (minimised by G):
      L_G = L_rec_rgb + HVI_weight * L_rec_hvi + L_adv_rgb + L_adv_hvi

      where each reconstruction term is:
          L_rec = L1 + SSIM + EdgeLoss + P_weight * PerceptualLoss

      and each adversarial term is:
          L_adv = adv_weight * LSGANGeneratorLoss(D(fake))

  Discriminator loss (minimised independently by D_rgb and D_hvi):
      L_D = LSGANDiscriminatorLoss(D(real), D(fake))

Training loop
-------------
  For every batch:
    1. Run G to get (output_rgb, output_hvi).
    2. Project GT to HVI space: gt_hvi = G.HVIT(gt_rgb).
    3. Update D_rgb: pass real=gt_rgb, fake=output_rgb.detach()
    4. Update D_hvi: pass real=gt_hvi, fake=output_hvi.detach()
    5. Update G: reconstruction losses + adversarial losses
                 (GAN signal is enabled only after `gan_start_epoch`).

Usage
-----
  python train_gan.py --dataset lol_v1 [other args from options_gan.py]
"""

import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.options_gan import option
from data.data import *
from data.scheduler import (
    CosineAnnealingRestartCyclicLR,
    CosineAnnealingRestartLR,
    GradualWarmupScheduler,
)
from eval import eval
from loss.losses import L1Loss, SSIM, EdgeLoss, PerceptualLoss
from loss.gan_losses import build_gan_losses
from measure import metrics
from net.HVI_GAN_Generator import HVIGANGenerator
from net.HVI_GAN_Discriminator import RGBDiscriminator, HVIDiscriminator

# --------------------------------------------------------------------------- #
# Parse arguments
# --------------------------------------------------------------------------- #
opt = option().parse_args()


# --------------------------------------------------------------------------- #
# Reproducibility helpers
# --------------------------------------------------------------------------- #
def seed_torch():
    seed = random.randint(1, 1_000_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if opt.gpu_mode and not torch.cuda.is_available():
        raise RuntimeError("No GPU found – run without --gpu_mode True.")


# --------------------------------------------------------------------------- #
# Dataset loader
# --------------------------------------------------------------------------- #
def load_datasets():
    print(f'===> Loading datasets: {opt.dataset}')
    if opt.dataset == 'lol_v1':
        train_set = get_lol_training_set(opt.data_train_lol_v1, size=opt.cropSize)
        test_set  = get_eval_set(opt.data_val_lol_v1)
    elif opt.dataset == 'lol_blur':
        train_set = get_training_set_blur(opt.data_train_lol_blur, size=opt.cropSize)
        test_set  = get_eval_set(opt.data_val_lol_blur)
    elif opt.dataset == 'lolv2_real':
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real, size=opt.cropSize)
        test_set  = get_eval_set(opt.data_val_lolv2_real)
    elif opt.dataset == 'lolv2_syn':
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn, size=opt.cropSize)
        test_set  = get_eval_set(opt.data_val_lolv2_syn)
    elif opt.dataset == 'SID':
        train_set = get_SID_training_set(opt.data_train_SID, size=opt.cropSize)
        test_set  = get_eval_set(opt.data_val_SID)
    elif opt.dataset == 'SICE_mix':
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        test_set  = get_SICE_eval_set(opt.data_val_SICE_mix)
    elif opt.dataset == 'SICE_grad':
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        test_set  = get_SICE_eval_set(opt.data_val_SICE_grad)
    elif opt.dataset == 'fivek':
        train_set = get_fivek_training_set(opt.data_train_fivek, size=opt.cropSize)
        test_set  = get_fivek_eval_set(opt.data_val_fivek)
    else:
        raise ValueError(f'Unknown dataset: {opt.dataset}')

    train_loader = DataLoader(
        train_set, num_workers=opt.threads,
        batch_size=opt.batchSize, shuffle=opt.shuffle,
    )
    test_loader = DataLoader(test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# Model builder
# --------------------------------------------------------------------------- #
def build_models():
    print('===> Building Generator + Discriminators')
    G       = HVIGANGenerator().cuda()
    D_rgb   = RGBDiscriminator().cuda()
    D_hvi   = HVIDiscriminator().cuda()

    if opt.start_epoch > 0:
        G_pth     = f'./weights/train/G_epoch_{opt.start_epoch}.pth'
        D_rgb_pth = f'./weights/train/D_rgb_epoch_{opt.start_epoch}.pth'
        D_hvi_pth = f'./weights/train/D_hvi_epoch_{opt.start_epoch}.pth'
        G.load_state_dict(torch.load(G_pth,     map_location='cuda'))
        D_rgb.load_state_dict(torch.load(D_rgb_pth, map_location='cuda'))
        D_hvi.load_state_dict(torch.load(D_hvi_pth, map_location='cuda'))
        print(f'  Loaded checkpoints from epoch {opt.start_epoch}')

    return G, D_rgb, D_hvi


# --------------------------------------------------------------------------- #
# Optimisers + schedulers
# --------------------------------------------------------------------------- #
def make_optimizers_and_schedulers(G, D_rgb, D_hvi):
    opt_G     = optim.Adam(G.parameters(),     lr=opt.lr,   betas=(0.9, 0.999))
    opt_D_rgb = optim.Adam(D_rgb.parameters(), lr=opt.lr_D, betas=(0.9, 0.999))
    opt_D_hvi = optim.Adam(D_hvi.parameters(), lr=opt.lr_D, betas=(0.9, 0.999))

    def _make_sched(optimizer, lr):
        if opt.cos_restart_cyclic:
            if opt.start_warmup:
                step = CosineAnnealingRestartCyclicLR(
                    optimizer=optimizer,
                    periods=[(opt.nEpochs // 4) - opt.warmup_epochs,
                              (opt.nEpochs * 3) // 4],
                    restart_weights=[1, 1],
                    eta_mins=[0.0002, 0.0000001],
                )
                return GradualWarmupScheduler(
                    optimizer, multiplier=1,
                    total_epoch=opt.warmup_epochs, after_scheduler=step,
                )
            return CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[opt.nEpochs // 4, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1],
                eta_mins=[0.0002, 0.0000001],
            )
        elif opt.cos_restart:
            n = opt.nEpochs - opt.start_epoch
            if opt.start_warmup:
                step = CosineAnnealingRestartLR(
                    optimizer=optimizer,
                    periods=[n - opt.warmup_epochs],
                    restart_weights=[1],
                    eta_min=1e-7,
                )
                return GradualWarmupScheduler(
                    optimizer, multiplier=1,
                    total_epoch=opt.warmup_epochs, after_scheduler=step,
                )
            return CosineAnnealingRestartLR(
                optimizer=optimizer, periods=[n],
                restart_weights=[1], eta_min=1e-7,
            )
        raise ValueError('Choose --cos_restart or --cos_restart_cyclic.')

    sched_G     = _make_sched(opt_G,     opt.lr)
    sched_D_rgb = _make_sched(opt_D_rgb, opt.lr_D)
    sched_D_hvi = _make_sched(opt_D_hvi, opt.lr_D)

    return (opt_G, opt_D_rgb, opt_D_hvi), (sched_G, sched_D_rgb, sched_D_hvi)


# --------------------------------------------------------------------------- #
# Loss initialisation
# --------------------------------------------------------------------------- #
def init_losses():
    """
    Returns a dict of all loss functions.
    Reconstruction losses are shared between the RGB and HVI domains.
    """
    rec = {
        'L1':  L1Loss(loss_weight=opt.L1_weight, reduction='mean').cuda(),
        'SSIM': SSIM(weight=opt.D_weight).cuda(),
        'Edge': EdgeLoss(loss_weight=opt.E_weight).cuda(),
        'Perc': PerceptualLoss(
            {'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
            perceptual_weight=1.0,
            criterion='mse',
        ).cuda(),
    }
    gan = build_gan_losses(
        d_weight_rgb=opt.D_rgb_weight,
        d_weight_hvi=opt.D_hvi_weight,
        g_weight_rgb=opt.adv_weight_rgb,
        g_weight_hvi=opt.adv_weight_hvi,
    )
    return rec, gan


# --------------------------------------------------------------------------- #
# Reconstruction loss helper
# --------------------------------------------------------------------------- #
def reconstruction_loss(pred, gt, losses):
    """
    L_rec = L1 + SSIM + Edge + P_weight * Perceptual
    Works identically for both RGB and HVI tensors.
    """
    l1   = losses['L1'](pred, gt)
    ssim = losses['SSIM'](pred, gt)
    edge = losses['Edge'](pred, gt)
    perc = losses['Perc'](pred, gt)[0]   # returns (percep, style)
    return l1 + ssim + edge + opt.P_weight * perc


# --------------------------------------------------------------------------- #
# Single training epoch
# --------------------------------------------------------------------------- #
def train_epoch(epoch, G, D_rgb, D_hvi, opt_G, opt_D_rgb, opt_D_hvi, rec_losses, gan_losses, training_data_loader):
    G.train(); D_rgb.train(); D_hvi.train()

    use_gan = (epoch >= opt.gan_start_epoch)
    loss_print = 0.0
    pic_cnt    = 0
    train_len  = len(training_data_loader)
    torch.autograd.set_detect_anomaly(opt.grad_detect)

    for i, batch in enumerate(tqdm(training_data_loader, desc=f'Epoch {epoch}')):
        im_low, im_gt, _, _ = batch
        im_low = im_low.cuda()
        im_gt  = im_gt.cuda()

        # Optional gamma augmentation
        if opt.gamma:
            gamma = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            output_rgb, output_hvi = G(im_low ** gamma)
        else:
            output_rgb, output_hvi = G(im_low)

        gt_rgb = im_gt
        gt_hvi = G.HVIT(gt_rgb)   # project ground truth into HVI space

        # ================================================================== #
        # 1.  Update Discriminators  (D_rgb and D_hvi)
        # ================================================================== #
        if use_gan:
            # --- D_rgb ---
            opt_D_rgb.zero_grad()
            pred_real_rgb = D_rgb(gt_rgb)
            pred_fake_rgb = D_rgb(output_rgb.detach())
            loss_D_rgb = gan_losses['D_rgb'](pred_real_rgb, pred_fake_rgb)
            loss_D_rgb.backward()
            if opt.grad_clip:
                torch.nn.utils.clip_grad_norm_(D_rgb.parameters(), 0.01)
            opt_D_rgb.step()

            # --- D_hvi ---
            opt_D_hvi.zero_grad()
            pred_real_hvi = D_hvi(gt_hvi.detach())
            pred_fake_hvi = D_hvi(output_hvi.detach())
            loss_D_hvi = gan_losses['D_hvi'](pred_real_hvi, pred_fake_hvi)
            loss_D_hvi.backward()
            if opt.grad_clip:
                torch.nn.utils.clip_grad_norm_(D_hvi.parameters(), 0.01)
            opt_D_hvi.step()

        # ================================================================== #
        # 2.  Update Generator
        # ================================================================== #
        opt_G.zero_grad()

        # Reconstruction losses in RGB space
        loss_rec_rgb = reconstruction_loss(output_rgb, gt_rgb, rec_losses)

        # Reconstruction losses in HVI space
        loss_rec_hvi = reconstruction_loss(output_hvi, gt_hvi, rec_losses)

        # Total reconstruction loss (mirrors original CIDNet weighting)
        loss_G = loss_rec_rgb + opt.HVI_weight * loss_rec_hvi

        # Adversarial losses (switched on after `gan_start_epoch`)
        if use_gan:
            pred_fake_rgb_G = D_rgb(output_rgb)          # no detach – G needs grads
            pred_fake_hvi_G = D_hvi(output_hvi)
            adv_rgb = gan_losses['G_rgb'](pred_fake_rgb_G)
            adv_hvi = gan_losses['G_hvi'](pred_fake_hvi_G)
            loss_G  = loss_G + adv_rgb + adv_hvi

        loss_G.backward()
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(G.parameters(), 0.01)
        opt_G.step()

        loss_print += loss_G.item()
        pic_cnt    += 1

        # ---- End-of-epoch logging ----------------------------------------
        if (i + 1) == train_len:
            lr_now = opt_G.param_groups[0]['lr']
            print(
                f"===> Epoch[{epoch}]: G_Loss={loss_print/pic_cnt:.4f} | "
                f"GAN={'ON' if use_gan else 'OFF'} | lr={lr_now}"
            )
            # Save a sample
            out_img = transforms.ToPILImage()(output_rgb[0].clamp(0, 1).squeeze(0))
            gt_img  = transforms.ToPILImage()(gt_rgb[0].squeeze(0))
            os.makedirs(opt.val_folder + 'training', exist_ok=True)
            out_img.save(opt.val_folder + 'training/sample.png')
            gt_img.save(opt.val_folder  + 'training/gt.png')

    return loss_print, pic_cnt


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
def checkpoint(epoch, G, D_rgb, D_hvi):
    os.makedirs('./weights/train', exist_ok=True)
    G_pth     = f'./weights/train/G_epoch_{epoch}.pth'
    D_rgb_pth = f'./weights/train/D_rgb_epoch_{epoch}.pth'
    D_hvi_pth = f'./weights/train/D_hvi_epoch_{epoch}.pth'
    torch.save(G.state_dict(),     G_pth)
    torch.save(D_rgb.state_dict(), D_rgb_pth)
    torch.save(D_hvi.state_dict(), D_hvi_pth)
    print(f'Checkpoints saved → epoch {epoch}')
    return G_pth


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == '__main__':

    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    G, D_rgb, D_hvi = build_models()
    (opt_G, opt_D_rgb, opt_D_hvi), (sched_G, sched_D_rgb, sched_D_hvi) = \
        make_optimizers_and_schedulers(G, D_rgb, D_hvi)
    rec_losses, gan_losses = init_losses()

    os.makedirs(opt.val_folder, exist_ok=True)
    os.makedirs('./results/training', exist_ok=True)

    start_epoch = opt.start_epoch
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')

    with open(f'./results/training/metrics_gan_{now}.md', 'w') as f:
        f.write(f'dataset: {opt.dataset}\n')
        f.write(f'lr_G: {opt.lr}  lr_D: {opt.lr_D}\n')
        f.write(f'batch_size: {opt.batchSize}  crop: {opt.cropSize}\n')
        f.write(f'HVI_weight: {opt.HVI_weight}  L1: {opt.L1_weight}  '
                f'SSIM: {opt.D_weight}  Edge: {opt.E_weight}  Perc: {opt.P_weight}\n')
        f.write(f'adv_rgb: {opt.adv_weight_rgb}  adv_hvi: {opt.adv_weight_hvi}  '
                f'gan_start_epoch: {opt.gan_start_epoch}\n')
        f.write('| Epoch | PSNR | SSIM | LPIPS |\n')
        f.write('|-------|------|------|-------|\n')

    psnr_log, ssim_log, lpips_log = [], [], []

    for epoch in range(start_epoch + 1, opt.nEpochs + start_epoch + 1):

        epoch_loss, pic_num = train_epoch(
            epoch,
            G, D_rgb, D_hvi,
            opt_G, opt_D_rgb, opt_D_hvi,
            rec_losses, gan_losses,
            training_data_loader,
        )

        # Step all schedulers together
        sched_G.step()
        sched_D_rgb.step()
        sched_D_hvi.step()

        if epoch % opt.snapshots == 0:
            G_pth = checkpoint(epoch, G, D_rgb, D_hvi)

            # ---- Determine eval paths (same as original train.py) ----------
            norm_size   = True
            is_lol_v1   = opt.dataset == 'lol_v1'
            is_lolv2_real = opt.dataset == 'lolv2_real'

            dataset_map = {
                'lol_v1':     ('LOLv1/',        opt.data_valgt_lol_v1),
                'lolv2_real': ('LOLv2_real/',    opt.data_valgt_lolv2_real),
                'lolv2_syn':  ('LOLv2_syn/',     opt.data_valgt_lolv2_syn),
                'lol_blur':   ('LOL_blur/',      opt.data_valgt_lol_blur),
                'SID':        ('SID/',           opt.data_valgt_SID),
                'SICE_mix':   ('SICE_mix/',      opt.data_valgt_SICE_mix),
                'SICE_grad':  ('SICE_grad/',     opt.data_valgt_SICE_grad),
                'fivek':      ('fivek/',         opt.data_valgt_fivek),
            }
            output_folder, label_dir = dataset_map[opt.dataset]
            if opt.dataset in ('SICE_mix', 'SICE_grad', 'fivek'):
                norm_size = False

            eval(
                G, testing_data_loader, G_pth,
                opt.val_folder + output_folder,
                norm_size=norm_size,
                LOL=is_lol_v1,
                v2=is_lolv2_real,
                alpha=0.8,
            )

            im_dir = opt.val_folder + output_folder + '*.png'
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)

            print(f'===> Avg.PSNR:  {avg_psnr:.4f} dB')
            print(f'===> Avg.SSIM:  {avg_ssim:.4f}')
            print(f'===> Avg.LPIPS: {avg_lpips:.4f}')

            psnr_log.append(avg_psnr)
            ssim_log.append(avg_ssim)
            lpips_log.append(avg_lpips)
            print('PSNR history:', psnr_log)
            print('SSIM history:', ssim_log)

            with open(f'./results/training/metrics_gan_{now}.md', 'a') as f:
                f.write(f'| {epoch} | {avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n')

        torch.cuda.empty_cache()
