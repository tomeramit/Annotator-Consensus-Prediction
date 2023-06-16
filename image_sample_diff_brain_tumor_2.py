"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import datetime
import json
import warnings
from pathlib import Path

import torch.distributed as dist
from mpi4py import MPI

from datasets.brain_tumor import BrainTumor2Dataset
from improved_diffusion import dist_util, logger
from improved_diffusion.sampling_util import sampling_major_vote_func
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.utils import set_random_seed

warnings.filterwarnings('ignore')


def main():
    args = create_argparser().parse_args()

    original_logs_path = Path(args.model_path).parent
    args.__dict__.update(json.loads((original_logs_path / 'args.json').read_text()))
    number_of_generated_instances = args.n_gen
    logs_path = original_logs_path / f"{Path(args.model_path).stem}_major_vote_{args.n_gen}"
    logger.info(args.__dict__)
    dist_util.setup_dist()
    
    logger.configure(dir=str(logs_path), log_suffix=f"val_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

    test_dataset=BrainTumor2Dataset(
        mode='val',
        image_size=args.image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        soft_label_gt=args.soft_label_training,
        consensus_gt=args.consensus_training
    )

    logger.log("creating model and diffusion...")


    args.condition_input_channel = 4

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    if args.__dict__.get("seed") is None:
        seed = 1234
    else:
        seed = int(args.__dict__.get("seed"))
    set_random_seed(seed, deterministic=True)
    logger.log("sampling major vote val")
    (logs_path / "major_vote").mkdir(exist_ok=True)
    step = int(Path(args.model_path).stem.split("_")[-1])

    sampling_major_vote_func(diffusion, model, str(logs_path / "major_vote"), test_dataset, logger, args.clip_denoised,
                             step=step, number_of_generated_instances=number_of_generated_instances)
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="/media/media1/shmuelsh/TomrCode/logs/2023-01-10-22-16-18-756357_brain_tumor_single_annotator_256_10_5e-05_1_100_0.0_0/ema_val_0.8067150_0.9999_035000.pt",
        n_gen=9,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
