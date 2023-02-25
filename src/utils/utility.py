from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torch
import json



def plot_imgs_tensor(x: torch.tensor, export_img, title: str = '', nrow=8, padding=2, normalize=False, pad_value=0):
    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()

    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)

    plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    plt.clf()


class Config(object):
    """Base class for experimental setting/configuration."""

    def __init__(self, settings):
        self.settings = settings

    def load_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            self.settings[key] = value

    def save_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w') as fp:
            json.dump(self.settings, fp)

