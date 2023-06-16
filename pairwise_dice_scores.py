import numpy as np

from datasets.brain_growth import BrainGrowthDataset
from datasets.brain_tumor import BrainTumor1Dataset
from datasets.kidney import KidneyDataset
from datasets.prostate import Prostate1Dataset, Prostate2Dataset


def dice_coefficient(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def calculate_dataset_scores(dataset):
    num_masks = len(dataset)
    scores = np.zeros((num_masks, num_masks))
    for i in range(num_masks):
        for j in range(i+1, num_masks):
            dice_score = dice_coefficient(dataset[i], dataset[j])
            scores[i, j] = dice_score
            scores[j, i] = dice_score
    return scores


def average_2d_array_without_diagonal(arr):
    # Create a boolean mask that identifies the elements on the main diagonal
    mask = np.eye(arr.shape[0], dtype=bool)

    # Use the mask to select the elements that are not on the main diagonal
    non_diagonal_elements = arr[~mask]

    # Calculate the average of the selected elements
    return np.mean(non_diagonal_elements)


# Example usage
for dataset in [("kidney", KidneyDataset(mode='train', no_annotators=True)),
                ("brain", BrainGrowthDataset(mode='train', no_annotators=True)),
                ("tumor", BrainTumor1Dataset(mode='train', no_annotators=True)),
                ("prostate1", Prostate1Dataset(mode='train', no_annotators=True)),
                ("prostate2", Prostate2Dataset(mode='train', no_annotators=True))]:

    masks_lists = dataset[1].case_name_to_mask_dict.values()

    average_pairwaise_dice_per_image = []
    for image_masks in list(masks_lists):
        scores_matrix = calculate_dataset_scores(image_masks)
        average_pairwaise_dice_per_image.append(average_2d_array_without_diagonal(scores_matrix))
    print(f"{dataset[0]}: {np.mean(average_pairwaise_dice_per_image)}")
