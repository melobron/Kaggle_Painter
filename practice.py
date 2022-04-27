import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from extract_modalities import ModalitiesEncoderTrainer


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr))]
    for m in members:
        tag = m.split('_')[1]
        train_writer.add_scalar(f"{tag}/{m}", getattr(trainer, m), iterations)


trainer = ModalitiesEncoderTrainer()
members = [attr for attr in dir(trainer)
           if ((not callable(getattr(trainer, attr))
                and not attr.startswith("__"))
               and ('loss' in attr
                    or 'grad' in attr
                    or 'nwd' in attr
                    or 'accuracy' in attr))]


