import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from datasets import TANH_MEAN, TANH_STD, inv_normalize
from models import Generator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path",
                        type=str,
                        # required=True,
                        help="Path to image")
    parser.add_argument("--checkpoint_model",
                        type=str,
                        default="saved_models_bk/netG-118000.pth",
                        help="Path to checkpoint model")
    parser.add_argument("--residual_blocks",
                        type=int,
                        default=23,
                        help="Number of residual blocks in G")
    opt = parser.parse_args()

    os.makedirs("images/outputs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model and load model checkpoint
    generator = Generator(filters=64,
                          num_res_blocks=opt.residual_blocks).to(device)
    generator.load_state_dict(torch.load(
        opt.checkpoint_model, map_location="cpu"))
    generator.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TANH_MEAN, TANH_STD)
    ])

    images = [opt.image_path, ] if os.path.isfile(opt.image_path) \
        else glob.glob(opt.image_path + "**/*.*", recursive=True)
    count = len(images)
    images = filter(os.path.isfile, images)

    for p in tqdm(images, total=count, desc="Inference"):
        try:
            image_tensor = (transform(Image.open(p).convert("RGB"))) \
                .unsqueeze(0) \
                .to(device)

            with torch.no_grad():
                sr_image = torch.clamp(generator(image_tensor) * 0.5 + 0.5, 0.0, 1.0) \
                    .permute(0, 2, 3, 1) \
                    .squeeze(0) \
                    .cpu() \
                    .numpy()
                assert sr_image.min() >= 0.0, sr_image.min()
                assert sr_image.max() <= 1.0, sr_image.max()

            # Save image
            fn = p.split("/")[-1]
            Image.fromarray((sr_image * 255).astype(np.uint8))\
                .save(f"images/outputs/{fn}")
        except Exception as e:
            tqdm.write(f"{p}, {str(e)}")
