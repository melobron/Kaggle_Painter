import torch
from tensorboardX import SummaryWriter

import argparse
import shutil

from extract_modalities import ModalitiesEncoderTrainer
from utils import *


import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def main(config, logger):
    print('Start extracting modalities \n')

    modalities_encoder_trainer = ModalitiesEncoderTrainer(config, logger)
    modalities_encoder_trainer.train()

    modalities_extraction_loader = ??








def setup(args):
    # Configure GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)

    # Load experiment setting
    config = get_config(args.config)

    # Setup directories
    output_directory = os.path.join(args.output_path, args.exp_name)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    checkpoint_directory, image_directory, logs_directory = make_result_folders(output_directory)
    writer = SummaryWriter(logs_directory)
    shutil.copy(args.config, os.path.join(output_directory, 'config.yaml'))

    # Setup config
    config['device'] = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    config['logger']['logs_dir'] = logs_directory
    config['logger']['checkpoint_dir'] = checkpoint_directory
    config['logger']['image_dir'] = image_directory

    return config, writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # GPU setting
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--use_cpu', type=bool, default=False)

    # Config file
    parser.add_argument('--config', type=str, default='./configs/dog2wolf.yaml')

    # Experiment setting
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--exp_name', type=str)

    args = parser.parse_args()

    config, logger = setup(args)
    main(config, logger)
