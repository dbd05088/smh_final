#!/usr/bin/env python3
from argparse import ArgumentParser
from pprint import pprint
import os
import resource
import random
import yaml
import torch
import colorful
import numpy as np
from tensorboardX import SummaryWriter
from data import DataScheduler
from spr_models.SPR import SPR
from train import train_model
from util import setup_logger, override_config

# Increase maximum number of open files from 1024 to 4096
# as suggested in https://github.com/pytorch/pytorch/issues/973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

class cls_SPR:

    def __init__(self, args, noisy_loader, unlabeled_loader, origin):

        self.noisy_loader = noisy_loader
        self.unlabeled_loader = unlabeled_loader
        self.origin = origin
        '''
        parser = ArgumentParser()
        
        parser.add_argument(
            '--random_seed', type=int, default=0)
        parser.add_argument(
            '--config', '-c', default='configs/ccmodel-coco.yaml'
        )
        parser.add_argument(
            '--episode', '-e', default='episodes/coco-split.yaml'
        )
        parser.add_argument('--log-dir', '-l')
        parser.add_argument('--resume-ckpt', default=None)
        parser.add_argument('--override', default='')

        args = parser.parse_args()
        '''
        logger = setup_logger()
        unique_identifier = "smh"

        # Load config
        config_path = args.configs
        episode_path = args.eps

        if args.resume_ckpt and not args.config:
            base_dir = os.path.dirname(os.path.dirname(args.resume_ckpt))
            config_path = os.path.join(base_dir, 'config.yaml')
            episode_path = os.path.join(base_dir, 'episode.yaml')
        self.config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        self.episode = yaml.load(open(episode_path), Loader=yaml.FullLoader)

        self.config['data_schedule'] = self.episode
        self.config['random_seed'] = args.random_seed
        if 'corruption_percent' not in self.config:
            self.config['corruption_percent'] = 0
        self.config = override_config(self.config, args.override)

        # Set log directory
        self.config['log_dir'] = os.path.join(args.log_dir, unique_identifier)
        if not args.resume_ckpt and os.path.exists(self.config['log_dir']):
            logger.warning('%s already exists' % self.config['log_dir'])
            #input('Press enter to continue')

        # print the configuration
        print(colorful.bold_white("configuration:").styled_string)
        print(self.config)
        print(colorful.bold_white("configuration end").styled_string)

        if args.resume_ckpt and not args.log_dir:
            self.config['log_dir'] = os.path.dirname(
                os.path.dirname(args.resume_ckpt)
            )

        # Save config
        os.makedirs(self.config['log_dir'], mode=0o755, exist_ok=True)
        if not args.resume_ckpt or args.config:
            config_save_path = os.path.join(self.config['log_dir'], 'config.yaml')
            episode_save_path = os.path.join(self.config['log_dir'], 'episode.yaml')
            yaml.dump(self.config, open(config_save_path, 'w'))
            yaml.dump(self.episode, open(episode_save_path, 'w'))
            print(colorful.bold_yellow('config & episode saved to {}'.format(
                self.config['log_dir'])).styled_string)

        # Build components
        if args.random_seed != 0:
            random.seed(args.random_seed)
            np.random.seed(args.random_seed)
            torch.manual_seed(args.random_seed)
        self.data_scheduler = DataScheduler(self.config)

        self.writer = SummaryWriter(self.config['log_dir'])
        self.model = SPR(self.config, self.writer)

        if args.resume_ckpt:
            model.load_state_dict(torch.load(args.resume_ckpt))
        self.model.to(self.config['device'])

        # data scheduler -> noisy loader로 change
        # train_model(self.config, self.model, self.data_scheduler, self.writer)


    def spr_train(self):
        train_model(self.config, self.model, self.data_scheduler, self.writer, self.unlabeled_loader, self.noisy_loader, self.origin)
        print(colorful.bold_pink("\nThank you and Good Job Computer").styled_string)
