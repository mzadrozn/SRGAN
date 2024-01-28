import os

import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from fire import Fire
from scripts.srganSR import SrganSR
from torchvision import transforms
from pathlib import Path
from scripts.dataloader import Set5Dataset, Set14Dataset
import tqdm
from skimage.metrics import structural_similarity


def main(config="test"):
    scale_factor = 4

    app = SrganSR(config)
    app.load()
    best_model = app.modelG
    best_model.eval()
    device = app.device
    test_dir = "tests" + f'\\X{scale_factor}' + "\\" + app.getModelName() + '\\Set14' + '\\'

    dataset = Set14Dataset(
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        lr_scale=scale_factor
    )

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    idx = 0
    names = ['img_001', 'img_002', 'img_005', 'img_011', 'img_014']

    for img, label in dataset:
        file = names[idx] + '.png'
        x = torch.unsqueeze(img,0)
        x = x.to(device)

        label = np.clip(torch.squeeze(label).cpu().detach().numpy(), 0, 1)
        label = ((label * 255) / np.max(label)).astype(np.uint8)

        output = best_model(x)
        output = np.clip(torch.squeeze(output).cpu().detach().numpy(), 0, 1)
        output = ((output * 255) / np.max(output)).astype(np.uint8)

        psnr = cv2.PSNR(output, label)
        print(f"Wartosc PSNR dla obrazu {file}, wynosi: {psnr}")
        test = np.moveaxis(output, 0, -1)

        img = Image.fromarray(test, 'RGB')
        img.save(test_dir + '\\' + file)
        idx = idx + 1


if __name__ == '__main__':
    Fire(main)