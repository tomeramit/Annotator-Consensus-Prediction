import numpy as np
from kornia import denormalize
from matplotlib import pyplot as plt

from datasets.QUBIC import QUBIKDataset
from datasets.multi_annotators_datasets import MultiAnnotatorsDataset
from datasets.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize, RandomAffine


class KidneyDataset(QUBIKDataset):
    def __init__(self, mode, no_aug=False,
                 image_size=256, shard=0, num_shards=1, consensus_gt=False, soft_label_gt=False, erosion=False, no_annotators=False):
        super().__init__(mode, no_aug, image_size, shard, num_shards, consensus_gt, soft_label_gt, erosion, no_annotators)
        if mode == 'train' and not no_aug:
            self.transformations = Compose([ToTensor(),
                                                        Resize(size=(image_size, image_size)),
                                                       # RandomVerticalFlip(), for kidney
                                                       RandomHorizontalFlip(),
                                                       RandomAffine(int(15), translate=(0.1, 0.1),  scale=(float(0.9), float(1.1))),

                                                       Normalize(self.mean, self.std)])

    def get_dataset_name(self):
        return "kidney"

    def get_task_num(self):
        return "1"

    @staticmethod
    def get_number_of_annotators():
        return 3

if __name__ == '__main__':
    mean = np.array([0])
    std = np.array([1])
    dataset = KidneyDataset('val', image_size=256, soft_label_gt=True)
    # dataset = KidneyDataset('train', image_size=320)

    for i in range(10):
        mask, out_dict, _ = dataset[i]
        img = out_dict["conditioned_image"]
        img_denorm = denormalize(img.unsqueeze(0), mean=dataset.mean, std=dataset.std)[0]

        plt.imshow(img_denorm.numpy().astype('float32'), cmap='gray')
        plt.show()

        mask = (mask[0] + 1) / 2
        plt.imshow(mask.numpy().astype('float32'), cmap='gray')
        plt.show()
        print('t')
