import numpy as np
from matplotlib import pyplot as plt

from datasets.QUBIC import QUBIKDataset
from datasets.multi_annotators_datasets import MultiAnnotatorsDataset


class BrainGrowthDataset(QUBIKDataset):
    def get_dataset_name(self):
        return "brain_growth"

    def get_task_num(self):
        return "1"

    @staticmethod
    def get_number_of_annotators():
        return 7

def test():
    dataset2 = BrainGrowthDataset('val', image_size=256, no_aug=True,output_specific_num_annotator_mask=True,erosion=True)
    dataset = BrainGrowthDataset('train', image_size=256,output_specific_num_annotator_mask=True,erosion=True)
    for k in range(len(dataset)):
        for alpha in [1, 2, 3, 4, 5, 6]:
            for truncated_p in [0.01, 0.99]:
                mask = dataset.mask_dict[dataset.image_names[k]]

                big_mask = (mask >= alpha)
                small_mask = (mask >= (alpha + 1))
                dataset.calc_annotation_mask(k, big_mask, small_mask, truncated_p)


    for k in range(len(dataset2)):
        for alpha in [1, 2, 3, 4, 5, 6]:
            for truncated_p in [0.01, 0.99]:
                mask = dataset.mask_dict[dataset.image_names[k]]

                big_mask = (mask >= alpha)
                small_mask = (mask >= (alpha + 1))
                dataset.calc_annotation_mask(k, big_mask, small_mask, truncated_p)

def test2():
    dataset = BrainGrowthDataset('train', image_size=256,output_specific_num_annotator_mask=True,erosion=True)
    for k in range(len(dataset)):
        mask = dataset.mask_dict[dataset.image_names[k]]
        for alpha in [1, 2, 3, 4, 5, 6]:
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
    mean = np.array([0])
    std = np.array([1])
    # dataset2 = BrainGrowthDataset('val', image_size=256, no_aug=True,output_specific_num_annotator_mask=True,erosion=True)
    dataset = BrainGrowthDataset('train', image_size=256, consensus_gt=False, output_specific_num_annotator_mask=True)

    for i in range(5):
        mask, out_dict, _ = dataset[i]
        img = out_dict["conditioned_image"]
        plt.imshow(img.permute(1,2,0).numpy().astype('float32'),cmap='gray')
        plt.show()

        plt.imshow(mask.permute(1,2,0).numpy().astype('float32'), cmap='gray')
        plt.show()

        mask, out_dict, _ = dataset2[i]
        img = out_dict["conditioned_image"]
        #plt.imshow(img.permute(1,2,0).numpy().astype('float32'),cmap='gray')
        #plt.show()
        #plt.imshow(mask.permute(1,2,0).numpy().astype('float32'), cmap='gray')
        #plt.show()