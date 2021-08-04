from Hang.utils_u_groupnorm_pytorchLightning import unetConv3d, unetUp3d, upsampleConv, concatConvUp, unetConv3dZ, unetUp3dZ
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import pdb

class unet3d(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 decay_factor = 0.2,
                feature_scale = 2, 
                n_classes = 6, 
                is_deconv = False, 
                in_channels = 6, 
                is_groupnorm = True,
                is_hpool = True,
                 weight_decay=0.01,
                weight = 0
    ):
        super(unet3d, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_groupnorm = is_groupnorm
        self.feature_scale = feature_scale
        self.weight = weight
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.weight_decay = weight_decay

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        #32, 64, 128, 256, 512

        # downsampling
        self.conv1 = unetConv3d(self.in_channels, filters[0], self.is_groupnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv2 = unetConv3d(filters[0], filters[1], self.is_groupnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv3 = unetConv3d(filters[1], filters[2], self.is_groupnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv4 = unetConv3d(filters[2], filters[3], self.is_groupnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        
        self.center = unetConv3d(filters[3], filters[4], self.is_groupnorm)
        
        # upsampling
        self.up_concat4 = unetUp3d(filters[4]+filters[3], filters[3], self.is_deconv) #remove this addition if is_deconv = true
        self.up_concat3 = unetUp3d(filters[3]+filters[2], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3d(filters[2]+filters[1], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d(filters[1]+filters[0], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1) 
        
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        up4 = self.up_concat4(conv4, self.center(maxpool4))
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
    
    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
#         criterion1 = nn.MSELoss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'loss': loss }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
#         criterion1 = nn.MSELoss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'val_loss': loss.cpu()}
        return {'val_loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss.cpu()}
        return {'avg_val_loss': avg_loss.cpu(), 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate) #, weight_decay = 0.1
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.decay_factor, patience=6),
            'monitor': 'avg_val_loss', 
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]