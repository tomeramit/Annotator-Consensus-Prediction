"""
Train a diffusion model on images.
"""

import argparse
import datetime
import json
import os
import warnings
from pathlib import Path

import git
from mpi4py import MPI

from datasets.brain_tumor import BrainTumor2Dataset
from datasets.multi_annotators_datasets import load_data
from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.utils import set_random_seed, set_random_seed_for_iterations

warnings.filterwarnings('ignore')


def main():
    args = create_argparser().parse_args()
    args.use_fp16 = True
    args.clip_denoised = True
    args.num_channels = 128
    args.image_size = 224
    args.num_res_blocks = 3
    args.learn_sigma = True
    args.deeper_net = True

    exp_name = f"brain_tumor_2_learn_sigma_erosion_floating_p_{args.rrdb_blocks}_{args.lr}_{args.batch_size}_{args.diffusion_steps}_{str(args.dropout)}_{args.weight_decay}_{args.n_gen}_{MPI.COMM_WORLD.Get_rank()}"

    logs_root = Path(__file__).absolute().parent.parent / "logs"
    log_path = logs_root / f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}_{exp_name}"
    os.environ["OPENAI_LOGDIR"] = str(log_path)
    set_random_seed(MPI.COMM_WORLD.Get_rank(), deterministic=True)
    set_random_seed_for_iterations(MPI.COMM_WORLD.Get_rank())
    dist_util.setup_dist()
    logger.configure(dir=str(log_path))

    if args.resume_checkpoint:
        resumed_checkpoint_arg = args.resume_checkpoint
        args.__dict__.update(json.loads((Path(args.resume_checkpoint) / 'args.json').read_text()))
        args.resume_checkpoint = resumed_checkpoint_arg

    args.condition_input_channel = 4
    if args.soft_label_training:
        args.number_of_annotators = None
    else:
        args.number_of_annotators = BrainTumor2Dataset.get_number_of_annotators()

    args.annotators_training = not args.consensus_training and not args.soft_label_training

    logger.info(args.__dict__)

    (Path(log_path) / 'args.json').write_text(json.dumps(args.__dict__, indent=4))
    logger.info(f"log folder path: {Path(log_path).resolve()}")

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    logger.log(f"git commit hash {sha}")

    logger.log("creating data loader...")
    data = load_data(
        dataset_class=BrainTumor2Dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        erosion=args.erosion,
        soft_label_training=args.soft_label_training,
        consensus_training=args.consensus_training
    )
    brain_dataset = BrainTumor2Dataset(
        mode='val',
        image_size=args.image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        soft_label_gt=True
    )

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"gpu {MPI.COMM_WORLD.Get_rank()} / {MPI.COMM_WORLD.Get_size()} validation images {len(brain_dataset)}")

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        clip_denoised=args.clip_denoised,
        logger=logger,
        image_size=args.image_size,
        val_dataset=brain_dataset,
        run_without_test=args.run_without_test,
        args=args
        # dist_util=dist_util,
    ).run_loop(max_iter=300000, start_print_iter=args.start_print_iter, number_of_generated_instances=args.n_gen)


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=0.00002,
        weight_decay=0.0,
        lr_anneal_steps=0,
        clip_denoised=False,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        save_interval=200,
        start_print_iter=600,
        log_interval=200,
        run_without_test=False,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        n_gen=9,
        erosion=True,
        condition_input_channel=4,
        soft_label_training=False,
        consensus_training=False,
        annotators_training=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
