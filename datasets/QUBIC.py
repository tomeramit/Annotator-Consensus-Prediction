from abc import ABC, abstractmethod

import nibabel as nib
import numpy as np

from datasets.multi_annotators_datasets import MultiAnnotatorsDataset


class QUBIKDataset(MultiAnnotatorsDataset, ABC):
    def __init__(self, mode, no_aug=False, image_size=256, shard=0, num_shards=1, consensus_gt=False, soft_label_gt=False, erosion=False, no_annotators=False):

        self.task_string = f"task0{self.get_task_num()}"
        super().__init__(mode, no_aug, image_size, shard, num_shards, consensus_gt, soft_label_gt, erosion, no_annotators=no_annotators)

    def get_val_case_name_to_image(self, dataset_path):
        case_name_to_image_dict = {}
        for case_path in (dataset_path / "val").iterdir():
            case_name_to_image_dict[f"{case_path.name}"] = nib.load(str(case_path / "image.nii.gz")).get_fdata()

        return case_name_to_image_dict

    def get_train_case_name_to_image(self, dataset_path):
        case_name_to_image_dict = {}
        for case_path in (dataset_path / "train").iterdir():
            case_name_to_image_dict[f"{case_path.name}"] = nib.load(str(case_path / "image.nii.gz")).get_fdata()

        return case_name_to_image_dict

    def get_train_case_name_to_masks(self, dataset_path):
        case_masks_dict = {}
        for case_path in (dataset_path / "train").iterdir():
            case_masks_dict[f"{case_path.name}"] = \
                np.stack([nib.load(str(mask_path)).get_fdata() for mask_path in sorted(list(case_path.glob(f"*{self.task_string}*")))])

        return case_masks_dict

    def get_val_case_name_to_masks(self, dataset_path):
        case_masks_dict = {}
        for case_path in (dataset_path / "val").iterdir():
            case_masks_dict[f"{case_path.name}"] = \
                np.stack([nib.load(str(mask_path)).get_fdata() for mask_path in sorted(list(case_path.glob(f"*{self.task_string}*")))])

        return case_masks_dict

    @staticmethod
    @abstractmethod
    def get_number_of_annotators():
        return NotImplemented

    @abstractmethod
    def get_dataset_name(self):
        return NotImplemented

    @abstractmethod
    def get_task_num(self):
        return NotImplemented
