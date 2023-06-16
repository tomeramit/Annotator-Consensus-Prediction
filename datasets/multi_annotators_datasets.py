import random
from abc import ABC, abstractmethod
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from mpi4py import MPI
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from datasets.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize, RandomVerticalFlip, RandomAffine
from improved_diffusion import logger

def load_data(
    *, batch_size, image_size, dataset_class, deterministic=False, erosion=False, soft_label_training=False, consensus_training=False, no_annotator_training=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """

    dataset = dataset_class(
        mode='train',
        image_size=image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        soft_label_gt=soft_label_training,
        consensus_gt=consensus_training,
        erosion=erosion,
        no_annotators=no_annotator_training
    )

    logger.log(f"gpu {MPI.COMM_WORLD.Get_rank()} / {MPI.COMM_WORLD.Get_size()} train images {len(dataset)}")

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader


class MultiAnnotatorsDataset(Dataset, ABC):

    CLASSES = ('brain-growth',)

    def __init__(self, mode, no_aug=False,
                 image_size=256, shard=0, num_shards=1, consensus_gt=False, soft_label_gt=False, erosion=False, no_annotators=False):

        self.consensus = consensus_gt
        self.no_annotators = no_annotators
        dataset_path = Path(__file__).absolute().parent.parent.parent / f"data/{self.get_dataset_name()}"
        self.soft_labeling = soft_label_gt
        self.number_of_annotators = self.get_number_of_annotators()
        self.erosion = erosion

        mean_list = []
        std_list = []
        self.case_name_to_image_dict = self.get_train_case_name_to_image(dataset_path)

        for image in self.case_name_to_image_dict.values():
            mean_list.append(image.mean(axis=(0, 1)))
            std_list.append(image.std(axis=(0, 1)))

        self.mean = torch.from_numpy(np.array(np.mean(mean_list, axis=0)))
        self.std = torch.from_numpy(np.array(np.mean(std_list, axis=0)))

        self.mode = mode

        if mode == 'train' and not no_aug:
            self.transformations = Compose([
                ToTensor(),
                Resize(size=(image_size, image_size)),
                RandomVerticalFlip(),
                RandomHorizontalFlip(),
                RandomAffine(int(15), translate=(0.1, 0.1),  scale=(float(0.9), float(1.1))),
                Normalize(self.mean, self.std)])
        else:
            self.transformations = Compose([
                ToTensor(),
                Resize(size=(image_size, image_size)),
                Normalize(self.mean, self.std)])

        if self.mode == 'train':
            self.case_name_to_mask_dict = self.get_train_case_name_to_masks(dataset_path)
        else:
            self.case_name_to_image_dict = self.get_val_case_name_to_image(dataset_path)
            self.case_name_to_mask_dict = self.get_val_case_name_to_masks(dataset_path)

        cases_to_remove = []
        for case_name, masks in self.case_name_to_mask_dict.items():
            if masks.shape[0] != self.number_of_annotators:
                logger.log(f"clearing {case_name} since it has only {masks.shape[0]} number of annotations {self.number_of_annotators} required")
                cases_to_remove.append(case_name)

        for case_name in cases_to_remove:
            del self.case_name_to_mask_dict[case_name]
            del self.case_name_to_image_dict[case_name]

        assert sum([self.soft_labeling, self.consensus, self.no_annotators, self.erosion]) == 1

        # consensus mode
        if self.consensus:
            for case_name, masks in self.case_name_to_mask_dict.items():
                self.case_name_to_mask_dict[case_name] = np.stack([np.sum(masks, axis=0) >= i for i in range(1, self.number_of_annotators + 1)])

        # soft labeling
        elif self.soft_labeling:
            for case_name, masks in self.case_name_to_mask_dict.items():
                self.case_name_to_mask_dict[case_name] = np.sum(masks, axis=0) / self.number_of_annotators

        else:
            # this mode is for raw gt - meaning each annotator
            pass

        self.image_names = sorted(list(self.case_name_to_image_dict.keys()))

        assert len(self.case_name_to_mask_dict.keys()) == len(self.case_name_to_image_dict.keys())

        self.max_len = len(self.image_names)
        if not self.mode == 'train':
            self.image_names = self.image_names[shard::num_shards]

    @staticmethod
    @abstractmethod
    def get_number_of_annotators():
        return NotImplemented

    @abstractmethod
    def get_val_case_name_to_image(self, dataset_path):
        return NotImplemented

    @abstractmethod
    def get_train_case_name_to_image(self, dataset_path):
        return NotImplemented

    @abstractmethod
    def get_train_case_name_to_masks(self, dataset_path):
        return NotImplemented

    @abstractmethod
    def get_val_case_name_to_masks(self, dataset_path):
        return NotImplemented

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        image_name = self.image_names[item]
        img = self.case_name_to_image_dict[image_name]
        mask = self.case_name_to_mask_dict[image_name]

        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = img[:, :, 0]

        if len(mask.shape) == 3 and mask.shape[-1] == 1:
            mask = mask[:, :, 0]

        if len(img.shape) == 4 and img.shape[-1] == 1:
            img = img[:, :, :, 0]

        if len(mask.shape) == 4 and mask.shape[-1] == 1:
            mask = mask[:, :, :, 0]

        out_dict = {}
        # if self.erosion:
        #     random_p = random.uniform(1, self.number_of_annotators)
        #     # random_p = random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
        #     if random_p == self.number_of_annotators:
        #         alpha = 6
        #         truncated_p = 1
        #     else:
        #         alpha = np.floor(random_p)
        #         truncated_p = random_p - alpha
        #     big_mask = (mask >= alpha)
        #     small_mask = (mask >= (alpha + 1))
        #     mask = self.calc_annotation_mask(item, big_mask, small_mask, truncated_p)
        #     out_dict["random_p"] = torch.tensor(random_p - alpha).type(torch.float32)

        if not self.soft_labeling:
            alpha = random.randint(1, self.number_of_annotators)
            mask = mask[alpha - 1]

            if self.consensus or self.erosion:
                out_dict["number_of_annotators"] = torch.tensor(alpha).type(torch.int32)

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
            img, mask = self.transformations(img.astype('float32'), mask.astype('float32'))
        else:
            img, mask = self.transformations(img.astype('float32'), mask.astype('float32'))
        out_dict["conditioned_image"] = img
        mask = (2 * mask - 1.0)

        return mask, out_dict, str(self.image_names[item])

    @staticmethod
    def calc_annotation_mask(item, big_mask, small_mask, truncated_p):
        desired_num_pixels = ((1 - truncated_p) * np.sum(big_mask) + (truncated_p * np.sum(small_mask))).astype(np.int)
        current_num_pixels = np.sum(big_mask)
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        current_mask = big_mask
        while current_num_pixels > desired_num_pixels:
            eroded_mask = cv.erode(current_mask.astype(np.float32), kernel, iterations=1)
            eroded_mask = np.logical_or(eroded_mask.astype(bool), small_mask.astype(bool))

            if np.sum(eroded_mask) == current_num_pixels:
                # plt.imshow(erode_mask, cmap='gray')
                # plt.title('erode_mask')
                # plt.show()
                #
                # plt.imshow(big_mask, cmap='gray')
                # plt.title('big_mask')
                # plt.show()
                #
                # plt.imshow(small_mask, cmap='gray')
                # plt.title('small_mask')
                # plt.show()

                # eroded_pixels_coords = ((current_mask.astype(int) - eroded_mask.astype(int)) > 0).nonzero()
                # exact_num_of_pixels_to_switch_on_current_mask = np.sum(current_mask) - desired_num_pixels  # positive
                # pixels_to_zero_on_current_mask = tuple([eroded_pixels_coords[0][:exact_num_of_pixels_to_switch_on_current_mask], eroded_pixels_coords[1][:exact_num_of_pixels_to_switch_on_current_mask]])
                # current_mask[pixels_to_zero_on_current_mask] = 0
                # eroded_mask = current_mask

                logger.log(f"stuck in erosion skipping image number {item} p {truncated_p} current_num_pixels {current_num_pixels} desired_num_pixels {desired_num_pixels}")
                current_mask = small_mask
                break

            if np.sum(eroded_mask) < desired_num_pixels:
                eroded_pixels_coords = ((current_mask.astype(int) - eroded_mask.astype(int)) > 0).nonzero()
                exact_num_of_pixels_to_switch_on_current_mask = np.sum(current_mask) - desired_num_pixels  # positive
                pixels_to_zero_on_current_mask = tuple([eroded_pixels_coords[0][:exact_num_of_pixels_to_switch_on_current_mask], eroded_pixels_coords[1][:exact_num_of_pixels_to_switch_on_current_mask]])
                current_mask[pixels_to_zero_on_current_mask] = 0
                eroded_mask = current_mask

            current_num_pixels = np.sum(eroded_mask)
            current_mask = eroded_mask

            # print("smaller number of pixels in mask: ", current_num_pixels, "instead of: ", desired_num_pixels)
        return current_mask

    @abstractmethod
    def get_dataset_name(self):
        return NotImplemented

