from pathlib import Path

import numpy as np
from kornia import denormalize
from matplotlib import pyplot as plt

from datasets.QUBIC import QUBIKDataset
from datasets.multi_annotators_datasets import MultiAnnotatorsDataset


class BrainTumor1Dataset(QUBIKDataset):
    def get_dataset_name(self):
        return "brain_tumor"

    def get_task_num(self):
        return "1"

    @staticmethod
    def get_number_of_annotators():
        return 3

class BrainTumor2Dataset(QUBIKDataset):
    def get_dataset_name(self):
        return "brain_tumor"

    def get_task_num(self):
        return "2"

    @staticmethod
    def get_number_of_annotators():
        return 3


class BrainTumor3Dataset(QUBIKDataset):
    def get_dataset_name(self):
        return "brain_tumor"

    def get_task_num(self):
        return "3"

    @staticmethod
    def get_number_of_annotators():
        return 3

def test():
    dataset2 = BrainTumor3Dataset('val', image_size=224, no_aug=True,output_specific_num_annotator_mask=True,erosion=True)
    dataset = BrainTumor3Dataset('train', image_size=224,output_specific_num_annotator_mask=True,erosion=True)
    for k in range(len(dataset)):
        for alpha in range(1, dataset.number_of_annotators):
            for truncated_p in [0.01, 0.99]:
                mask = dataset.mask_dict[dataset.image_names[k]]

                big_mask = (mask >= alpha)
                small_mask = (mask >= (alpha + 1))
                dataset.calc_annotation_mask(k, big_mask, small_mask, truncated_p)


    for k in range(len(dataset2)):
        for alpha in range(1, dataset.number_of_annotators):
            for truncated_p in [0.01, 0.99]:
                mask = dataset.mask_dict[dataset.image_names[k]]

                big_mask = (mask >= alpha)
                small_mask = (mask >= (alpha + 1))
                dataset.calc_annotation_mask(k, big_mask, small_mask, truncated_p)

def test2():
    dataset = BrainTumor3Dataset('train', image_size=224,output_specific_num_annotator_mask=True,erosion=True)
    for k in range(len(dataset)):
        mask = dataset.mask_dict[dataset.image_names[k]]
        for alpha in range(1, dataset.number_of_annotators):
            big_mask = (mask >= alpha)
            small_mask = (mask >= (alpha + 1))
            plt.imshow(big_mask.astype('float32'), cmap='gray')
            plt.show()
            for truncated_p in [0, 0.2,0.4,0.7, 1]:
                out_mask = dataset.calc_annotation_mask(k, big_mask, small_mask, truncated_p)
                plt.imshow(out_mask.astype('float32'), cmap='gray')
                plt.show()
            plt.imshow(small_mask.astype('float32'), cmap='gray')
            plt.show()
            print('t')


def test3():
    import nibabel as nib
    path_for_case = Path("/media/media1/shmuelsh/TomrCode/data/brain_tumor/val/case30/")
    img_path = path_for_case / "image.nii.gz"

    img = nib.load(str(img_path)).get_fdata()

    plt.imshow(img[:,:,0], cmap='gray')
    plt.show()

    plt.imshow(img[:,:,1], cmap='gray')
    plt.show()

    plt.imshow(img[:,:,2], cmap='gray')
    plt.show()

    plt.imshow(img[:,:,3], cmap='gray')
    plt.show()
    a = sorted(list(path_for_case.glob("*task*")))
    for mask_path in a:
        mask = nib.load(str(mask_path)).get_fdata()
        plt.imshow(mask, cmap='gray')
        plt.show()

    print('t')

    # mask = nib_mask.get_fdata()

def test4():
    import nibabel as nib
    path_for_case = Path("/media/media1/shmuelsh/TomrCode/data/brain_tumor/train/case04/")
    img_path = path_for_case / "image.nii.gz"

    img = nib.load(str(img_path)).get_fdata()

    plt.imshow(img[:,:,0], cmap='gray')
    plt.show()

    plt.imshow(img[:,:,1], cmap='gray')
    plt.show()

    plt.imshow(img[:,:,2], cmap='gray')
    plt.show()

    plt.imshow(img[:,:,3], cmap='gray')
    plt.show()
    a = sorted(list(path_for_case.glob("*task*")))
    for mask_path in a:
        mask = nib.load(str(mask_path)).get_fdata()
        plt.imshow(mask, cmap='gray')
        plt.show()

    print('t')

    # mask = nib_mask.get_fdata()

def test5():
    mean = np.array([0])
    std = np.array([1])
    dataset = BrainTumor3Dataset('val', image_size=224, no_aug=True,output_specific_num_annotator_mask=False,erosion=False)
    # dataset = BrainTumor3Dataset('train', image_size=224,no_aug=True)
    logs_path = Path("/media/media1/tomeramit/logs/temp/val")
    logs_path.mkdir(exist_ok=True)
    for i in range(len(dataset)):
        mask, out_dict, _ = dataset[i]
        img = out_dict["conditioned_image"]
        img_denorm = denormalize(img.unsqueeze(0), mean=dataset.mean, std=dataset.std)[0]
        # plt.imshow(img_denorm[0].numpy().astype('float32'), cmap='gray')
        # plt.show()
        #
        # plt.imshow(img_denorm[1].numpy().astype('float32'), cmap='gray')
        # plt.show()
        #
        # plt.imshow(img_denorm[2].numpy().astype('float32'), cmap='gray')
        # plt.show()

        plt.imshow(img_denorm[3].numpy().astype('float32'), cmap='gray')
        # plt.show()
        plt.savefig(logs_path / f"{str(i)}_img")
        plt.close()

        mask = (mask[0] + 1) / 2
        plt.imshow(mask.numpy().astype('float32'), cmap='gray')
        # plt.show()
        plt.savefig(logs_path / f"{str(i)}_mask")
        plt.close()

        # mask, out_dict, _ = dataset2[i]
        # img = out_dict["conditioned_image"]
        #plt.imshow(img.permute(1,2,0).numpy().astype('float32'),cmap='gray')
        #plt.show()
        #plt.imshow(mask.permute(1,2,0).numpy().astype('float32'), cmap='gray')
        #plt.show()

if __name__ == '__main__':
    # test()
    # test2()
    # exit(0)
    # test4()
    mean = np.array([0])
    std = np.array([1])
    # dataset2 = BrainTumor3Dataset('val', image_size=224, no_aug=True,output_specific_num_annotator_mask=False,erosion=False)
    dataset = BrainTumor3Dataset('train', image_size=224,no_aug=True)

    for i in range(5):
        mask, out_dict, _ = dataset[i]
        img = out_dict["conditioned_image"]
        img_denorm = denormalize(img.unsqueeze(0), mean=dataset.mean, std=dataset.std)[0]
        # plt.imshow(img_denorm[0].numpy().astype('float32'), cmap='gray')
        # plt.show()
        #
        # plt.imshow(img_denorm[1].numpy().astype('float32'), cmap='gray')
        # plt.show()
        #
        # plt.imshow(img_denorm[2].numpy().astype('float32'), cmap='gray')
        # plt.show()

        plt.imshow(img_denorm[3].numpy().astype('float32'), cmap='gray')
        plt.show()

        mask = (mask[0] + 1) / 2
        plt.imshow(mask.numpy().astype('float32'), cmap='gray')
        plt.show()

        # mask, out_dict, _ = dataset2[i]
        # img = out_dict["conditioned_image"]
        #plt.imshow(img.permute(1,2,0).numpy().astype('float32'),cmap='gray')
        #plt.show()
        #plt.imshow(mask.permute(1,2,0).numpy().astype('float32'), cmap='gray')
        #plt.show()