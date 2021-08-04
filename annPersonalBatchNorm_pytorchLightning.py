from Hang.utils_u_groupnorm_pytorchLightning import unetConv3d, unetUp3d, upsampleConv, concatConvUp
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import pdb

class ann_256_32(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 filters,
                 decay_factor = 0.2,
                 n_classes = 6, 
                 in_channels = 6
    ):
        super(ann_256_32, self).__init__()
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor

        self.in1 = nn.Linear(in_channels,256)
        self.h1 = nn.Linear(256,32)
        self.out = nn.Linear(32,n_classes)
        self.prelu = nn.PReLU()

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, inputs):
        x = self.prelu(self.bn1(self.in1(inputs)))
        x = self.prelu(self.bn2(self.h1(x)))
        x = self.out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
#         criterion1 = nn.L1Loss()
        criterion1 = nn.MSELoss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'loss': loss }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
#         criterion1 = nn.L1Loss()
        criterion1 = nn.MSELoss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'val_loss': loss.cpu()}
        return {'val_loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss.cpu()}
        return {'avg_val_loss': avg_loss.cpu(), 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.decay_factor, patience=6),
            'monitor': 'avg_val_loss', 
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
    
    
class ann_256_256_32(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 filters,
                 decay_factor = 0.2,
                 n_classes = 6, 
                 in_channels = 6
    ):
        super(ann_256_256_32, self).__init__()
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor

        self.in1 = nn.Linear(in_channels,256)
        self.h1 = nn.Linear(256,256)
        self.h2 = nn.Linear(256,32)
        self.out = nn.Linear(32,n_classes)
        self.prelu = nn.PReLU()

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, inputs):
        x = self.prelu(self.bn1(self.in1(inputs)))
        x = self.prelu(self.bn2(self.h1(x)))
        x = self.prelu(self.bn3(self.h2(x)))
        x = self.out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
#         criterion1 = nn.L1Loss()
        criterion1 = nn.MSELoss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'loss': loss }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
#         criterion1 = nn.L1Loss()
        criterion1 = nn.MSELoss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'val_loss': loss.cpu()}
        return {'val_loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss.cpu()}
        return {'avg_val_loss': avg_loss.cpu(), 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.decay_factor, patience=6),
            'monitor': 'avg_val_loss', 
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
class ann_32_256_256_32(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 filters,
                 decay_factor = 0.2,
                 n_classes = 6, 
                 in_channels = 6
    ):
        super(ann_32_256_256_32, self).__init__()
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor

        self.in1 = nn.Linear(in_channels,32)
        self.h1 = nn.Linear(32,256)
        self.h2 = nn.Linear(256,256)
        self.h3 = nn.Linear(256, 32)
        self.out = nn.Linear(32,n_classes)
        self.prelu = nn.PReLU()

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(32)

    def forward(self, inputs):
        x = self.prelu(self.bn1(self.in1(inputs)))
        x = self.prelu(self.bn2(self.h1(x)))
        x = self.prelu(self.bn3(self.h2(x)))
        x = self.prelu(self.bn4(self.h3(x)))
        x = self.out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
#         criterion1 = nn.L1Loss()
        criterion1 = nn.MSELoss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'loss': loss }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
#         criterion1 = nn.L1Loss()
        criterion1 = nn.MSELoss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'val_loss': loss.cpu()}
        return {'val_loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss.cpu()}
        return {'avg_val_loss': avg_loss.cpu(), 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.decay_factor, patience=6),
            'monitor': 'avg_val_loss', 
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]