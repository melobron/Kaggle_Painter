
from dataloaders.data import EncoderDataset
from models.modalities_encoder import ModalitiesEncoder


class ModalitiesEncoderTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.model = ModalitiesEncoder(config).to(config['device'])
        self.




    def train(self, epoch):



