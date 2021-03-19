import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from datasets import ImageDataset, inv_normalize
from models import Discriminator, FeatureExtractor, Generator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",
                        type=int,
                        default=0,
                        help="epoch to start training from")

    parser.add_argument("--netG_checkpoint",
                        type=str,
                        default="",
                        help="generator weights checkpoint file")
    parser.add_argument("--netD_checkpoint",
                        type=str,
                        default="",
                        help="discriminator weights checkpoint file")

    parser.add_argument("--hr_height",
                        type=int,
                        default=192,
                        help="high res. image height")
    parser.add_argument("--hr_width",
                        type=int,
                        default=192,
                        help="high res. image width")
    parser.add_argument("--residual_blocks",
                        type=int,
                        default=32,
                        help="number of residual blocks in the generator")

    parser.add_argument("--n_epochs",
                        type=int,
                        default=1000,
                        help="number of epochs of training")
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="size of the batches")
    parser.add_argument("--n_cpu",
                        type=int,
                        default=16,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--multi_gpu",
                        action="store_true",
                        help="multi-gpu training")
    parser.add_argument("--mixed_precision",
                        action="store_true",
                        help="mixed precision training")

    parser.add_argument("--lr_pretrain",
                        type=float,
                        default=2e-4,
                        help="pretrain learning rate")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="adam: learning rate")
    parser.add_argument("--b1",
                        type=float,
                        default=0.9,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2",
                        type=float,
                        default=0.999,
                        help="adam: decay of first order momentum of gradient")

    parser.add_argument("--sample_interval",
                        type=int,
                        default=2000,
                        help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval",
                        type=int,
                        default=2000,
                        help="batch interval between model checkpoints")

    parser.add_argument("--warmup_batches",
                        type=int,
                        default=3000,
                        help="number of batches for the pretrain stage")
    parser.add_argument("--lambda_adv",
                        type=float,
                        default=5e-3,
                        help="adversarial loss weight")
    parser.add_argument("--lambda_pixel",
                        type=float,
                        default=1e-2,
                        help="pixel-wise loss weight")

    opt = parser.parse_args()
    # print(opt)

    ###########################################################################

    # Init

    os.makedirs("saved_models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter()

    # Get models

    hr_shape = (opt.hr_height, opt.hr_width)
    generator = Generator(3, filters=64,
                          num_res_blocks=opt.residual_blocks) \
        .to(device).train()
    discriminator = Discriminator() \
        .to(device).train()
    feature_extractor = FeatureExtractor() \
        .to(device).eval()

    if opt.netG_checkpoint:
        try:
            generator.load_state_dict(torch.load(opt.netG_checkpoint,
                                                 map_location="cpu"))
            print(
                f"[x] Restored generator weights from: {opt.netG_checkpoint}")
        except:
            print("[!] Generator weights from scratch.")
    if opt.netD_checkpoint:
        try:
            discriminator.load_state_dict(torch.load(opt.netD_checkpoint,
                                                     map_location="cpu"))
            print(
                f"[x] Restored discriminator weights from: {opt.netD_checkpoint}")
        except:
            print("[!] Discriminator weights from scratch.")

    # Get optimizers and lr schedulers

    optimizer_G_pretrain = optim.Adam(generator.parameters(), lr=opt.lr_pretrain,
                                      betas=(opt.b1, opt.b2))
    optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr,
                             betas=(opt.b1, opt.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr,
                             betas=(opt.b1, opt.b2))

    milestones = [50000, 100000, 200000, 300000]
    gamma = 0.5
    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G,
                                                 milestones=milestones,
                                                 gamma=gamma)
    scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D,
                                                 milestones=milestones,
                                                 gamma=gamma)

    # Mixed precision and multi-GPU training

    if opt.mixed_precision:
        print("Mixed precision training")
        scaler = GradScaler()

    if opt.multi_gpu and torch.cuda.device_count() > 1:
        print("Multi-GPU training")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # Get data

    dataloader = DataLoader(
        ImageDataset("../datasets/DIV2K", hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True
    )

    # Get criteria

    criterion_pixel = nn.L1Loss().to(device)
    criterion_content = nn.L1Loss().to(device)
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)

    # Training

    global_steps = 0

    for epoch in tqdm(range(opt.epoch, opt.n_epochs), desc="Training"):

        for i, imgs in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):

            imgs_lr = imgs["lr"].to(device)
            imgs_hr = imgs["hr"].to(device)

            # Train generator

            optimizer_G.zero_grad()

            with autocast(opt.mixed_precision):
                # Generate a high resolution image from low resolution input
                gen_hr = generator(imgs_lr)

                # Measure pixel-wise loss against ground truth
                loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            if global_steps < opt.warmup_batches:
                # Warm-up (pixel-wise loss only)

                if opt.mixed_precision:
                    scaler.scale(loss_pixel).backward()
                    scaler.step(optimizer_G_pretrain)
                    scaler.update()
                else:
                    loss_pixel.backward()
                    optimizer_G_pretrain.step()

                tqdm.write(
                    "[%09d] [G pixel: %.4f]"
                    % (global_steps, loss_pixel.item())
                )

                global_steps += 1
                continue

            with autocast(opt.mixed_precision):
                pred_real = discriminator(imgs_hr)
                pred_fake = discriminator(gen_hr)

                # Adversarial loss (relativistic average GAN)
                loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True),
                                         torch.ones_like(pred_fake))

                # Content loss
                gen_features = feature_extractor(gen_hr)
                real_features = feature_extractor(imgs_hr)
                loss_content = criterion_content(gen_features, real_features)

                # Total generator loss
                loss_G = loss_content \
                    + opt.lambda_adv * loss_GAN \
                    + opt.lambda_pixel * loss_pixel

            if opt.mixed_precision:
                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)
                scaler.update()
            else:
                loss_G.backward()
                optimizer_G.step()
            scheduler_G.step()

            # Train discriminator

            optimizer_D.zero_grad()

            with autocast(opt.mixed_precision):
                pred_real = discriminator(imgs_hr)
                pred_fake = discriminator(gen_hr.detach())

                tqdm.write(f"pred_real.shape: {pred_real.shape}")

                loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True),
                                          torch.ones_like(pred_real))
                loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True),
                                          torch.zeros_like(pred_fake))

                loss_D = loss_real + loss_fake

            if opt.mixed_precision:
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                scaler.update()
            else:
                loss_D.backward()
                optimizer_D.step()
            scheduler_D.step()

            writer.add_scalar("lossD/sum", loss_D.item(),
                              global_steps)
            writer.add_scalar("lossD/real", loss_real.item(),
                              global_steps)
            writer.add_scalar("lossD/fake", loss_fake.item(),
                              global_steps)
            writer.add_scalar("lossG/sum", loss_G.item(),
                              global_steps)
            writer.add_scalar("lossG/content", loss_content.item(),
                              global_steps)
            writer.add_scalar("lossG/adversarial", loss_GAN.item(),
                              global_steps)
            writer.add_scalar("lossG/pixel", loss_pixel.item(),
                              global_steps)

            tqdm.write(
                "[%09d] [D loss: %.4f, real: %.4f, fake: %.4f] [G loss: %.4f, content: %.4f, adv: %.4f, pixel: %.4f]"
                % (
                    global_steps,

                    loss_D.item(),
                    loss_real.item(),
                    loss_fake.item(),

                    loss_G.item(),
                    loss_content.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                )
            )

            global_steps += 1
            if global_steps % opt.sample_interval == 0:
                writer.add_images("gen_hr", inv_normalize(torch.clamp(gen_hr, -1, 1)),
                                  global_steps)
                writer.add_images("imgs_lr", inv_normalize(F.interpolate(imgs_lr, scale_factor=4)),
                                  global_steps)
                writer.add_images("imgs_hr", inv_normalize(imgs_hr),
                                  global_steps)

            if global_steps % opt.checkpoint_interval == 0:
                torch.save(generator.module.state_dict() if hasattr(generator, "module") else generator.state_dict(),
                           f"saved_models/netG-{global_steps}.pth")
                torch.save(discriminator.module.state_dict() if hasattr(discriminator, "module") else discriminator.state_dict(),
                           f"saved_models/netD-{global_steps}.pth")

    print("Done!")
