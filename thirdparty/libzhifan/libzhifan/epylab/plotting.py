import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def eimshow(img,
           *args,
           **kwargs):
    title = kwargs.pop('title', f"image size = {img.shape}")
    if isinstance(img, np.ndarray):
        # img = np.squeeze(img)
        plt.imshow(img)
        plt.title(title)
    elif HAS_TORCH:
        if isinstance(img, torch.Tensor):
            old_shape = img.shape
            img = img.squeeze().detach().cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            if img.shape != old_shape:
                title = f"squeeze shape: {old_shape} \n=> {img.shape}"
            eimshow(img, *args, **kwargs, title=title)
    else:  # Fall back
        plt.imshow(img, *args, **kwargs)

def eimsave(img, path='tmp.png'):
    eimshow(img)
    plt.savefig(path)
