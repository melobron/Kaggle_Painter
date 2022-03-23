import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class ModalitiesEncoder(nn.Module):
    def __init__(self, config):
        super(ModalitiesEncoder, self).__init__()

        resnet18 = models.resnet18(pretrained=False)
        num_feat = resnet18.fc.in_features
        self.feat_extractor = nn.Sequential(*list(resnet18.children())[:-1])

        # MLP
        self.MLP = nn.Sequential(*[
            nn.Linear(num_feat, num_feat),
            nn.ReLU(inplace=True),
            nn.Linear(num_feat, config['modalities_encoder']['out_dim'])
        ])

        # NT-Xent loss
        self.nt_xent_loss = NTXentLoss(config['device'], config['modalities_encoder']['batch_size'],
                                       **config['modalities_encoder']['loss'])

    def forward(self, x):
        h = self.feat_extractor(x)
        z = self.MLP(h)
        return h, z

    def calc_nt_xent_loss(self, xis, xjs):
        his, zis = self.forward(xis)  # batch * C
        hjs, zjs = self.forward(xjs)  # batch * C

        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_loss(zis, zjs)
        return loss


class NTXentLoss(nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.temperature = temperature

        self.similarity_function = self.get_similarity_function(use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            return self.cosine_similarity
        else:
            return self.dot_similarity

    def get_correlated_mask(self):
        diag = np.eye(2*self.batch_size)
        l1 = np.eye((2*self.batch_size), 2*self.batch_size, k=-self.batch_size)
        l2 = np.eye((2*self.batch_size), 2*self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1-mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def dot_similarity(x, y):
        v = torch.tensordot(x.unsqueeze(dim=1), y.unsqueeze(dim=0), dims=2)  # (2N, 1, C) x (1, 2N, C) = (2N, 2N)
        return v

    @staticmethod
    def cosine_similarity(self, x, y):
        v = nn.CosineSimilarity(x.unsqueeze(dim=1), y.unsqueeze(dim=0))  # (2N, 1, C) x (1, 2N, C) = (2N, 2N)
        return v

    def forward(self, zis, zjs):
        representation = torch.cat([zis, zjs], dim=0)
        similarity_matrix = self.similarity_function(representation, representation)  # (2N, 2N)

        # Scores for the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2*self.batch_size, 1)  # (2N, 1)

        # Scores for the negative samples
        negatives = similarity_matrix[self.get_correlated_mask()].view(2*self.batch_size, -1)  # (2N, 2N-2)

        logits = torch.cat([positives, negatives], dim=1)  # (2N, 2N-1)
        logits /= self.temperature

        labels = torch.zeros(2*self.batch_size).to(self.device)
        loss = self.criterion(logits, labels)

        return loss / (2*self.batch_size)
