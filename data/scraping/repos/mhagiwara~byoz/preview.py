from struct import pack
import sys
import random

import cv2
import numpy as np
import torch

from dalle_pytorch import OpenAIDiscreteVAE, DALLE
from torch.serialization import save

vae = OpenAIDiscreteVAE()


def to_numpy(x):
    x = x.permute(0, 2, 3, 1).squeeze(0)
    x = x.clamp(min=0.) * 255
    x = x.cpu().detach().numpy()

    return x


def save_image(image, file_path):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(file_path, image)


def pack_images(array, ncols):
    nindex, height, width, channels = array.shape
    nrows = nindex // ncols
    assert nindex == nrows*ncols
    result = (array.reshape(nrows, ncols, height, width, channels)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, channels))
    return result


def main():
    random.seed(41)

    zs = []
    for line in sys.stdin:
        z = line.strip().split(' ')[2:]
        z = torch.tensor([[int(v) for v in z]])
        zs.append(z)

    images = []
    zs = random.sample(zs, k=min(40, len(zs)))
    for z in zs:
        im_recon = vae.decode(z)
        im_recon = to_numpy(im_recon)

        images.append(im_recon)
    
    image = pack_images(np.array(images), ncols=4)
    save_image(image, 'recon.png')

if __name__ == '__main__':
    main()
