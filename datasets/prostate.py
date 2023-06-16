import numpy as np
from kornia import denormalize
from matplotlib import pyplot as plt

from datasets.QUBIC import QUBIKDataset
from datasets.multi_annotators_datasets import MultiAnnotatorsDataset


class Prostate1Dataset(QUBIKDataset):
    def get_dataset_name(self):
        return "prostate"

    def get_task_num(self):
        return "1"

    @staticmethod
    def get_number_of_annotators():
        return 6

class Prostate2Dataset(QUBIKDataset):
    def get_dataset_name(self):
        return "prostate"

    def get_task_num(self):
        return "2"

    @staticmethod
    def get_number_of_annotators():
        return 6


def test():
    dataset2 = Prostate1Dataset('val', image_size=480, no_aug=True,output_specific_num_annotator_mask=True,erosion=True)
    dataset = Prostate1Dataset('train', image_size=480,output_specific_num_annotator_mask=True,erosion=True)
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
    dataset = Prostate1Dataset('train', image_size=480,output_specific_num_annotator_mask=True,erosion=True)
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


if __name__ == '__main__':
    # test()
    # test2()
    # exit(0)
    # test4()
    mean = np.array([0])
    std = np.array([1])
    dataset2 = Prostate1Dataset('val', image_size=480, no_aug=True,output_specific_num_annotator_mask=False,erosion=False)
    # dataset = Prostate2Dataset('train', image_size=480,output_specific_num_annotator_mask=True,erosion=True)

    for i in range(5):
        mask, out_dict, _ = dataset2[i]
        img = out_dict["conditioned_image"]
        img_denorm = denormalize(img.unsqueeze(0), mean=dataset2.mean, std=dataset2.std)[0]
        plt.imshow(img_denorm[0].numpy().astype('float32'), cmap='gray')
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