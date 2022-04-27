import torch
import torch.optim as optim

from dataloaders.data import EncoderDataset
from models.modalities_encoder import ModalitiesEncoder
from utils import *


class ModalitiesEncoderTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = config['device']

        self.model = ModalitiesEncoder(config).to(self.device)
        self.epochs = self.config['modalities_encoder']['epochs']
        self.batch_size = self.config['modalities_encoder']['batch_size']
        self.lr = config['modalities_encoder']['lr']
        self.weight_decay = config['modalities_encoder']['weight_decay']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.data_loader = EncoderDataset(config).get_dataloader()
        self.T_max = len(self.data_loader.dataset)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.T_max, eta_min=0)

    def train(self):
        for epoch in range(1, self.epochs+1):
            with Timer('Elapsed time for epoch: {}'):
                for batch, (xis, xjs) in enumerate(self.data_loader):
                    self.optimizer.zero_grad()

                    xis, xjs = xis.to(self.device), xjs.to(self.device)
                    loss_enc_contrastive = self.model.calc_nt_xent_loss(xis, xjs)
                    loss_enc_contrastive.backward()
                    self.optimizer.step()

                    print("Epoch: {curr_epoch}/{total_epochs} | Progress: {progress}% | Loss: {curr_loss}"
                          .format(curr_epoch=epoch, total_epochs=self.epochs,
                                  progress=100.*batch/len(self.data_loader),
                                  curr_loss=loss_enc_contrastive))

                    # Logging loss
                    if (batch + 1) % self.config['logger']['log_loss'] == 0:
                        write_loss(batch, self, self.logger)

                if epoch % self.config['logger']['checkpoint_modalities_encoder_every'] == 0:
                    self.save(self.config['logger']['checkpoint_dir'], epoch)

                if epoch >= 10:
                    self.scheduler.step()

    def save(self, checkpoint_dir, epoch):
        name = os.path.join(checkpoint_dir, 'enc_{epoch}.pt'.format(epoch=str(epoch)))
        encoder = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(encoder, name)


class ModalitiesExtractor:
    def __init__(self, config):
        self.config = config

    def extract_embeddings(self, encoder, data_loader):

        encoder.eval()
        for


        return

    def cluster_embeddings(self, ?, k, ?):

        return

    def get_modalities(self, encoder, data_loader):

        return

    def get_modalities_grid_image(self, modalities, images_per_modality=10):



        return


