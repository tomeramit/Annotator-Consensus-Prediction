import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as tvu
from PIL import Image
from kornia import denormalize
from matplotlib import pyplot as plt
from mpi4py import MPI
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader

from datasets.QUBIC import QUBIKDataset
from datasets.disc_region import DiscRegionDataset
from datasets.monu import MonuDataset
from datasets.multi_annotators_datasets import MultiAnnotatorsDataset
from . import dist_util
from .metrics import FBound_metric, WCov_metric
from .qubiq_metrics import qubiq_metric
from .utils import set_random_seed_for_iterations

cityspallete = [
    0, 0, 0,
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]


def calculate_metrics(x, gt):
    predict = x.detach().cpu().numpy().astype('uint8')
    target = gt.detach().cpu().numpy().astype('uint8')
    return f1_score(target.flatten(), predict.flatten()), jaccard_score(target.flatten(), predict.flatten()), \
           WCov_metric(predict, target), FBound_metric(predict, target)


def sampling_major_vote_func(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, step, number_of_generated_instances=9):
    ddp_model.eval()
    batch_size = 1
    major_vote_number = number_of_generated_instances
    logger.info(f"number of images to visualize {len(dataset)}")
    loader = DataLoader(dataset, batch_size=batch_size)
    name_unique_prefix = MPI.COMM_WORLD.Get_rank()
    f1_score_list = []
    soft_dice_score_list_9 = []
    soft_dice_score_list_5 = []
    miou_list = []
    fbound_list = []
    wcov_list = []

    with torch.no_grad():
        for index, (gt_mask, condition_on, name) in enumerate(loader):

            if isinstance(dataset, QUBIKDataset):
                set_random_seed_for_iterations(step + int(name[0].split("_")[0][-2:]))
            elif isinstance(dataset, DiscRegionDataset):
                set_random_seed_for_iterations(step + int(name[0].split("_")[-1][5:]))
            else:
                set_random_seed_for_iterations(step + int(name[0].split("_")[1]))
            gt_mask = (gt_mask + 1.0) / 2.0
            condition_on_image = condition_on["conditioned_image"]
            former_frame_for_feature_extraction = condition_on_image.to(dist_util.dev())

            for i in range(gt_mask.shape[0]):
                cm = plt.get_cmap('bwr')
                gt_img = Image.fromarray((cm((gt_mask[i][0].detach().cpu().numpy()))[:, :, :3] * 255).astype(np.uint8))
                gt_img.save(
                    os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_gt_colormap.png"))

            if isinstance(dataset, MultiAnnotatorsDataset):
                annotator_gt = gt_mask * dataset.number_of_annotators
                for i in range(dataset.number_of_annotators):
                    gt_img = Image.fromarray((annotator_gt >= (i + 1))[0][0].detach().cpu().numpy().astype('uint8'))
                    gt_img.putpalette(cityspallete)
                    gt_img.save(
                        os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_gt_annotator_{i + 1}_palette.png"))

            for i in range(condition_on_image.shape[0]):
                denorm_condition_on = denormalize(condition_on_image.clone(), mean=dataset.mean, std=dataset.std)
                # denorm_condition_mask = denorm_condition_on.clone()
                # denorm_condition_mask[denorm_condition_mask <= 20] = 0
                # denorm_condition_mask[denorm_condition_mask > 20] = 1
                if isinstance(dataset, MultiAnnotatorsDataset):
                    tvu.save_image(
                        denorm_condition_on[i,] / denorm_condition_on.max(),
                        os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_condition_on_image.png")
                    )
                else:
                    tvu.save_image(
                        denorm_condition_on[i,] / 255.,
                        os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_condition_on_image.png")
                    )

            if isinstance(dataset, MonuDataset):
                _, _, W, H = former_frame_for_feature_extraction.shape
                kernel_size = dataset.image_size
                stride = 256
                patches = []
                for y, x in np.ndindex((((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1)):
                    y = y * stride
                    x = x * stride
                    patches.append(former_frame_for_feature_extraction[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)])
                patches = torch.stack(patches)

                major_vote_list = []
                for i in range(major_vote_number):
                    x_list = []

                    for index in range(math.ceil(patches.shape[0] / 4)):
                        model_kwargs = {"conditioned_image": patches[index * 4: min((index + 1) * 4, patches.shape[0])]}
                        x = diffusion_model.p_sample_loop(
                                ddp_model,
                                (model_kwargs["conditioned_image"].shape[0], gt_mask.shape[1], model_kwargs["conditioned_image"].shape[2], model_kwargs["conditioned_image"].shape[3]),
                                progress=True,
                                clip_denoised=clip_denoised,
                                model_kwargs=model_kwargs
                            )

                        x_list.append(x)
                    out = torch.cat(x_list)

                    output = torch.zeros((former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2], former_frame_for_feature_extraction.shape[3]))
                    idx_sum = torch.zeros((former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2], former_frame_for_feature_extraction.shape[3]))
                    for index, val in enumerate(out):
                        y, x = np.unravel_index(index, (((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1))
                        y = y * stride
                        x = x * stride

                        idx_sum[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)] += 1

                        output[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)] += val[:, :min(y + kernel_size, W) - y, :min(x + kernel_size, H) - x].cpu().data.numpy()

                    output = output / idx_sum
                    major_vote_list.append(output)

                x = torch.cat(major_vote_list)
            elif isinstance(dataset, MultiAnnotatorsDataset):
                model_kwargs = {
                    "inference": True,
                    "first_time_step": diffusion_model.num_timesteps
                }

                if ddp_model.soft_label_training:
                    model_kwargs["conditioned_image"] = torch.cat([former_frame_for_feature_extraction])
                    first_dim = 1
                    # model_kwargs["number_of_annotators"] = None

                else: #consensus or annotators
                    if not ddp_model.no_annotator_training:
                        model_kwargs["number_of_annotators"] = torch.range(1, dataset.number_of_annotators, dtype=torch.int).to(dist_util.dev())
                    model_kwargs["conditioned_image"] = torch.cat([former_frame_for_feature_extraction] * dataset.number_of_annotators)
                    first_dim = dataset.number_of_annotators

                list_of_x = []
                for i in range(number_of_generated_instances):
                    x = diffusion_model.p_sample_loop(
                        ddp_model,
                        (first_dim, gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                         former_frame_for_feature_extraction.shape[3]),
                        progress=True,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs
                    )
                    list_of_x.append(x)
                x = torch.stack(list_of_x)
            else:
                model_kwargs = {
                    "conditioned_image": torch.cat([former_frame_for_feature_extraction] * major_vote_number)}

                x = diffusion_model.p_sample_loop(
                    ddp_model,
                    (major_vote_number, gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                     former_frame_for_feature_extraction.shape[3]),
                    progress=True,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs
                )

            x = (x + 1.0) / 2.0

            np.save(os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_result_array"), x.detach().cpu().numpy())
            np.save(os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_gt_mask_array"), gt_mask.detach().cpu().numpy())

            if x.shape[-1] != gt_mask.shape[-1] or x.shape[-2] != gt_mask.shape[-2]:
                x = F.interpolate(x, gt_mask.shape[2:], mode='bilinear')

            x = torch.clamp(x, 0.0, 1.0)
            cm = plt.get_cmap('bwr')
            if isinstance(dataset, MultiAnnotatorsDataset):
                for gen_im in range(x.shape[0]):
                    for annotator in range(x.shape[1]):
                        out_img = Image.fromarray(x[gen_im][annotator][0].round().detach().cpu().numpy().astype('uint8'))
                        out_img.putpalette(cityspallete)
                        out_img.save(
                            os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_annotator_{annotator+1}_vote_{gen_im}_binary.png"))

                        out_img = Image.fromarray((cm((x[gen_im][annotator][0].detach().cpu().numpy()))[:, :, :3] * 255).astype(np.uint8))
                        out_img.save(
                            os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_annotator_{annotator+1}_vote_{gen_im}_colormap.png"))

            # major vote result
            x = x.mean(dim=0)

            for i in range(x.shape[0]):
                out_img = Image.fromarray(x[i][0].round().detach().cpu().numpy().astype('uint8'))
                out_img.putpalette(cityspallete)
                out_img.save(
                    os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_major_vote_annotator_{i+1}.png"))

                cm = plt.get_cmap('bwr')
                out_img = Image.fromarray((cm((x[i][0].detach().cpu().numpy()))[:, :, :3] * 255).astype(np.uint8))
                out_img.save(
                    os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_major_vote_annotator_{i+1}_colormap.png"))

            if isinstance(dataset, MultiAnnotatorsDataset) and not ddp_model.soft_label_training and not ddp_model.no_annotator_training:
                x = x.round()
                if ddp_model.consensus_training:
                    annotator_gt = gt_mask * dataset.number_of_annotators
                    for i in range(dataset.number_of_annotators):
                        out_im = x[i].int()
                        single_annotator = (annotator_gt >= i+1).int()

                        f1, miou, wcov, fbound = calculate_metrics(out_im[0], single_annotator[0][0])

                        logger.info(
                            f"{name_unique_prefix}_{index}_{name[0]} annotator {i+1} iou {miou}, f1_Score {f1}, WCov {wcov}, boundF {fbound}")

                x = torch.sum(x, dim=0) / dataset.number_of_annotators

                out_img = Image.fromarray((cm((x[0].detach().cpu().numpy()))[:, :, :3] * 255).astype(np.uint8))
                out_img.save(
                    os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_final_colormap.png"))

            if ddp_model.soft_label_training:
                x = x[0]

            if ddp_model.no_annotator_training:
                x = x.round()
                x = x.mean(dim=0)
                out_img = Image.fromarray((cm((x[0].detach().cpu().numpy()))[:, :, :3] * 255).astype(np.uint8))
                out_img.save(
                    os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_final_colormap.png"))

            for i, (gt_eval, out_im) in enumerate(zip(gt_mask, x)):

                if isinstance(dataset, MultiAnnotatorsDataset):
                    b = out_im.unsqueeze(0).detach().cpu() # * denorm_condition_mask
                    # b[b<0.5] = 0 #need to make this as hyper parameter
                    avg_dice, dice_score_list = qubiq_metric(b.unsqueeze(0), gt_eval.unsqueeze(0).detach().cpu(), num_of_thresholds=9)
                    soft_dice_score_list_9.append(avg_dice)

                    logger.info(
                        f"{name_unique_prefix}_{index}_{name[0]} soft dice 9 {soft_dice_score_list_9[-1]}")

                    logger.info(
                        f"{name_unique_prefix}_{index}_{name[0]} soft dice list {dice_score_list}")

                    avg_dice, dice_score_list = qubiq_metric(b.unsqueeze(0), gt_eval.unsqueeze(0).detach().cpu(), num_of_thresholds=5)
                    soft_dice_score_list_5.append(avg_dice)

                    logger.info(
                        f"{name_unique_prefix}_{index}_{name[0]} soft dice 5 {soft_dice_score_list_5[-1]}")

                    logger.info(
                        f"{name_unique_prefix}_{index}_{name[0]} soft dice list {dice_score_list}")

                out_im = out_im.round().int()
                gt_eval = gt_eval.round().int()

                f1, miou, wcov, fbound = calculate_metrics(out_im, gt_eval[0])
                f1_score_list.append(f1)
                miou_list.append(miou)
                wcov_list.append(wcov)
                fbound_list.append(fbound)

                logger.info(
                    f"{name_unique_prefix}_{index}_{name[0]} iou {miou_list[-1]}, f1_Score {f1_score_list[-1]}, WCov {wcov_list[-1]}, boundF {fbound_list[-1]}")

    logger.info(f"waiting for rest of the processes for barrier 1")

    dist.barrier()

    logger.info(f"passing barrier 1")
    my_length = len(dataset)
    max_single_len = int(np.ceil(dataset.max_len / dist.get_world_size()))
    logger.info(f"{my_length} {max_single_len}")
    iou_tensor = torch.tensor(miou_list + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    f1_tensor = torch.tensor(f1_score_list + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    wcov_tensor = torch.tensor(wcov_list + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    boundf_tensor = torch.tensor(fbound_list + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    soft_dice_tensor_5 = torch.tensor(soft_dice_score_list_5 + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    soft_dice_tensor_9 = torch.tensor(soft_dice_score_list_9 + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    gathered_miou = [torch.ones_like(iou_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_f1 = [torch.ones_like(f1_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_wcov = [torch.ones_like(wcov_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_boundf = [torch.ones_like(boundf_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_soft_dice_5 = [torch.ones_like(soft_dice_tensor_5) * -1 for _ in range(dist.get_world_size())]
    gathered_soft_dice_9 = [torch.ones_like(soft_dice_tensor_9) * -1 for _ in range(dist.get_world_size())]

    logger.info(f"Iou tensor{gathered_miou}")
    logger.info(f"Iou tensor{gathered_f1}")
    logger.info(f"Iou tensor{gathered_wcov}")
    logger.info(f"Iou tensor{gathered_boundf}")
    logger.info(f"Iou tensor{gathered_soft_dice_5}")
    logger.info(f"Iou tensor{gathered_soft_dice_9}")

    logger.info(f"Iou tensor{iou_tensor}")
    logger.info(f"Iou tensor{f1_tensor}")
    logger.info(f"Iou tensor{wcov_tensor}")
    logger.info(f"Iou tensor{boundf_tensor}")
    logger.info(f"Iou tensor{soft_dice_tensor_5}")
    logger.info(f"Iou tensor{soft_dice_tensor_9}")

    dist.all_gather(gathered_miou, iou_tensor)
    dist.all_gather(gathered_f1, f1_tensor)
    dist.all_gather(gathered_wcov, wcov_tensor)
    dist.all_gather(gathered_boundf, boundf_tensor)
    dist.all_gather(gathered_soft_dice_5, soft_dice_tensor_5)
    dist.all_gather(gathered_soft_dice_9, soft_dice_tensor_9)

    # if dist.get_rank() == 0:
    logger.info("measure total avg")
    logger.info(f"Iou tensor{iou_tensor}")
    logger.info(f"{gathered_miou}")

    gathered_miou = torch.cat(gathered_miou)
    logger.info(f"1")
    gathered_miou = gathered_miou[gathered_miou != -1]
    logger.info(f"mean iou {gathered_miou.mean()}")

    logger.info(f"2")
    gathered_f1 = torch.cat(gathered_f1)
    gathered_f1 = gathered_f1[gathered_f1 != -1]
    logger.info(f"mean f1 {gathered_f1.mean()}")
    gathered_wcov = torch.cat(gathered_wcov)
    gathered_wcov = gathered_wcov[gathered_wcov != -1]
    logger.info(f"mean WCov {gathered_wcov.mean()}")
    gathered_boundf = torch.cat(gathered_boundf)
    gathered_boundf = gathered_boundf[gathered_boundf != -1]
    logger.info(f"mean boundF {gathered_boundf.mean()}")

    gathered_soft_dice_9 = torch.cat(gathered_soft_dice_9)
    gathered_soft_dice_9 = gathered_soft_dice_9[gathered_soft_dice_9 != -1]
    logger.info(f"mean soft dice 9 {gathered_soft_dice_9.mean()}")

    gathered_soft_dice_5 = torch.cat(gathered_soft_dice_5)
    gathered_soft_dice_5 = gathered_soft_dice_5[gathered_soft_dice_5 != -1]
    logger.info(f"mean soft dice 5 {gathered_soft_dice_5.mean()}")

    logger.info(f"waiting for rest of the processes for barrier 2")
    dist.barrier()
    logger.info(f"passing barrier 2")
    return gathered_soft_dice_5.mean().item()
