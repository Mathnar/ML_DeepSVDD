from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms


class CIFAR10_data(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_cl = tuple([normal_class])
        self.outlier_cl = list(range(0, 10))
        self.outlier_cl.remove(normal_class)

        # Preprocessed min and max values
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]] * 3,
                                                             [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_cl))

        training_set = MyCIFAR10(root=self.root, train=True, download=True,
                              transform=transform, target_transform=target_transform)
        training_idx_normal = get_target_label_idx(training_set.targets, self.normal_cl)
        self.training_set = Subset(training_set, training_idx_normal)

        self.testing_set = MyCIFAR10(root=self.root, train=False, download=True,
                                     transform=transform, target_transform=target_transform)



class MyCIFAR10(CIFAR10):#Torchvision CIFAR10 ritorna index e datasample
    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
