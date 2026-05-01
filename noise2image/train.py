"""
noise2image.train — Lightning wrapper around the U-Net reconstruction model.

Defines the :class:`Model` LightningModule that wraps
:class:`noise2image.models.unet_attention.Unet` (default) or
:class:`noise2image.models.resunet.ResUnet` (``vanilla_unet=True``
fallback), plus the :class:`MultiDataloaderEarlyStopping` callback used
to drive early stopping off multiple validation dataloaders.

Adapted from the public Noise2Image codebase (Cao et al.) with these
modifications for Noise2Params:
  * 2-channel (positive, negative) event input instead of the original
    single-channel stream.
  * FiLM time-conditioning on a fixed integration-time scalar (paper: 5 s).
  * The MultiDataloaderEarlyStopping callback, so early stopping can be
    driven by a combination of validation-loss signals when training
    with more than one validation set (synthetic vs. real).

This file is imported by :mod:`noise2params.noise_image` at inference
time (``from noise2image.train import Model``) and by
:file:`train_synthetic_6.py` at training time.

Note on naming: ``train.py`` is the upstream Noise2Image training entry
point; our active training entry point is ``train_synthetic_6.py``,
which imports :class:`Model` and :class:`MultiDataloaderEarlyStopping`
from here.  Running ``python train.py`` directly is not the intended
workflow in this release.
"""
import os
from argparse import ArgumentParser
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as l
from lightning.pytorch import loggers
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
import torchmetrics

try:
    from .models.unet_attention import Unet
    from .models.resunet import ResUnet
except:
    from models.unet_attention import Unet
    from models.resunet import ResUnet
# import np_transforms
# import utils
try:
    from . import np_transforms
    from . import utils
except:
    import utils
    import np_transforms

INDIST_EVENT_PATH = './data/indist_events/'
INDIST_IMAGE_PATH = './data/indist_images/'
OOD_EVENT_PATH = './data/ood_DIV2K_events/'
OOD_IMAGE_PATH = './data/ood_DIV2K_images/'

parser = ArgumentParser()
parser.add_argument("--gpu_ind", type=int, default=0, help="GPU index")
parser.add_argument("--vanilla_unet", action='store_true', 
                    help="Use vanilla U-Net instead of the advanced u-net with attention layers")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=3, help="Batch size")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader")
parser.add_argument("--log_name", type=str, default='', 
                    help="Name of the log & checkpoint folder under ./lightning_logs.")
parser.add_argument("--checkpoint_path", type=str, default='', 
                    help="Path to the checkpoint to load from. Will skip training if provided.")
parser.add_argument("--time_bin", type=int, default=1, 
                    help="Time binning during the event aggregation. Note that default value is 1, which means all events are aggregated into a single time bin.")
parser.add_argument("--pixel_bin", type=int, default=2, 
                    help="Pixel binning during the event aggregation.")
parser.add_argument("--polarity", action='store_true', 
                    help="Aggregate events into 2 channels for positive and negative polarities.")
parser.add_argument("--time_std", action='store_true', 
                    help="Add a channel for standard deviation of the timestamp.")
parser.add_argument("--integration_time_s", type=float, default=1, 
                    help="Event aggregation time in seconds.")

torch.set_float32_matmul_precision('medium')


class Model(l.LightningModule):
    def __init__(self, dim, in_channels, lr, vanilla_unet=False, ):
        super().__init__()

        if vanilla_unet:
            self.model = ResUnet(in_channels=in_channels, out_channels=1, dim=dim, conv_kernel_size=3)
        else:
            self.model = Unet(
                dim=dim,
                dim_mults=(1, 2, 4, 8), #default
                # dim_mults=(1, 2, 4, 8, 16),
                in_channels=in_channels,
                out_channels=1,
                flash_attn=True,
                # flash_attn=False,
            )

        self.lr = lr
        # self.running_sum = 0

        self.valid_metrics = torchmetrics.MetricCollection({
            'valid_psnr': torchmetrics.image.PeakSignalNoiseRatio(),
            'valid_ssim': torchmetrics.image.StructuralSimilarityIndexMeasure()
        })
        self.test_metrics = torchmetrics.MetricCollection({
            'test_psnr': torchmetrics.image.PeakSignalNoiseRatio(),
            'test_ssim': torchmetrics.image.StructuralSimilarityIndexMeasure()
        })
        self.save_hyperparameters()

    def forward(self, x, time):
        return self.model(x, time=time)

    def on_train_epoch_start(self):
        # self.running_sum = 0
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        # self.running_sum = 0
        return super().on_validation_epoch_start()

    def training_step(self, batch, batch_idx):
        x, y, t = batch
        y_hat = self.model(x, t)
        loss = (y_hat - y).pow(2).mean()
        # self.running_sum += loss.item()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('avg_train_loss', self.running_sum / (batch_idx + 1), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y, t = batch
    #     y_hat = self.model(x, time=t)
    #     loss = (y_hat - y).pow(2).mean()
    #     self.running_sum += loss.item()
    #     self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #
    #     self.valid_metrics(y_hat, y)
    #     self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #
    #     if batch_idx == 0:
    #         self.logger.experiment.add_images('noise', np.tile((torch.sum(x, dim=1, keepdim=True).cpu()), (1, 3, 1, 1)),
    #                                           self.current_epoch)
    #         self.logger.experiment.add_images('reconstruction', np.tile((y_hat.cpu()), (1, 3, 1, 1)),
    #                                           self.current_epoch)
    #         self.logger.experiment.add_images('truth', np.tile((y.cpu()), (1, 3, 1, 1)), self.current_epoch)
    #     self.log('avg_val_loss', self.running_sum / (batch_idx + 1), on_step=True, on_epoch=False, prog_bar=True)
    #     return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, t = batch
        y_hat = self.model(x, time=t)
        loss = (y_hat - y).pow(2).mean()

        # Log with simple names - Lightning adds dataloader_idx suffix automatically
        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

        # Update metrics and log with simple names
        self.valid_metrics(y_hat, y)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Image logging still needs manual dataloader distinction
        if batch_idx == 0:
            self.logger.experiment.add_images(f'noise_dl{dataloader_idx}',
                                              np.tile((torch.sum(x, dim=1, keepdim=True).cpu()), (1, 3, 1, 1)),
                                              self.current_epoch)
            self.logger.experiment.add_images(f'reconstruction_dl{dataloader_idx}',
                                              np.tile((y_hat.cpu()), (1, 3, 1, 1)),
                                              self.current_epoch)
            self.logger.experiment.add_images(f'truth_dl{dataloader_idx}',
                                              np.tile((y.cpu()), (1, 3, 1, 1)),
                                              self.current_epoch)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, t = batch
        y_hat = self.model(x, time=t)
        loss = (y_hat - y).pow(2).mean()
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)

        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 2:
            x, t = batch
        else:
            x, y, t = batch
        return self.model(x, t)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99))
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    #     return {
    #         'optimizer': optimizer,
    #         'lr_scheduler': scheduler,
    #         'monitor': 'val_loss'
    #     }
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Full name with Lightning's suffix
                "interval": "epoch",
                "frequency": 1,
            },
        }

# class Model(l.LightningModule):
#     def __init__(self, dim, in_channels, lr, vanilla_unet=False):
#         super().__init__()
#
#         if vanilla_unet:
#             self.model = ResUnet(in_channels=in_channels, out_channels=1, dim=dim, conv_kernel_size=3)
#         else:
#             self.model = Unet(
#                 dim=dim,
#                 dim_mults=(1, 2, 4, 8),
#                 in_channels=in_channels,
#                 out_channels=1,
#                 flash_attn=True,
#             )
#
#         self.lr = lr
#
#         # Metrics expect floating tensors; keep their internal dtype policy simple
#         self.valid_metrics = torchmetrics.MetricCollection({
#             "valid_psnr": torchmetrics.image.PeakSignalNoiseRatio(),
#             "valid_ssim": torchmetrics.image.StructuralSimilarityIndexMeasure()
#         })
#         self.test_metrics = torchmetrics.MetricCollection({
#             "test_psnr": torchmetrics.image.PeakSignalNoiseRatio(),
#             "test_ssim": torchmetrics.image.StructuralSimilarityIndexMeasure()
#         })
#         self.save_hyperparameters()
#
#     def forward(self, x, time):
#         return self.model(x, time=time)
#
#     # ----------------------
#     # helpers: dtype-safe logging
#     # ----------------------
#     @staticmethod
#     def _to_float_for_metrics(*tensors):
#         # TorchMetrics commonly expects float tensors in [0,1] or a known range.
#         # We only change dtype here; if you need range normalization, do it upstream.
#         out = []
#         for t in tensors:
#             # Detach to avoid autograd clutter in logging
#             tt = t.detach()
#             if tt.dtype in (torch.float16, torch.bfloat16):
#                 tt = tt.float()
#             out.append(tt)
#         return out if len(out) > 1 else out[0]
#
#     @staticmethod
#     def _make_rgb_for_tb(t):
#         """
#         t: (N, C, H, W) torch tensor, possibly half/bfloat16
#         Returns float32, 3-channel tensor on CPU for add_images.
#         """
#         t = t.detach()
#         if t.dtype in (torch.float16, torch.bfloat16):
#             t = t.float()
#         if t.shape[1] == 1:
#             t = t.repeat(1, 3, 1, 1)
#         # Optional clamp if your images are in [0,1]; remove if your range differs
#         t = t.clamp(0, 1)
#         return t.cpu()
#
#     # ----------------------
#     # training / validation / test
#     # ----------------------
#     def training_step(self, batch, batch_idx):
#         x, y, t = batch
#         y_hat = self.model(x, time=t)
#         loss = (y_hat - y).pow(2).mean()
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx, dataloader_idx=0):
#         x, y, t = batch
#         y_hat = self.model(x, time=t)
#         loss = (y_hat - y).pow(2).mean()
#
#         # Loss
#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#
#         # Metrics: cast to float32 for stability and dtype compatibility
#         y_hat_f32, y_f32 = self._to_float_for_metrics(y_hat, y)
#         self.valid_metrics(y_hat_f32, y_f32)
#         self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#
#         # Image logging without NumPy
#         if batch_idx == 0 and hasattr(self.logger, "experiment"):
#             # example "noise" as sum over event channels -> single channel
#             noise_1ch = torch.sum(x, dim=1, keepdim=True)
#             self.logger.experiment.add_images(
#                 f"noise_dl{dataloader_idx}",
#                 self._make_rgb_for_tb(noise_1ch),
#                 global_step=self.current_epoch,
#             )
#             self.logger.experiment.add_images(
#                 f"reconstruction_dl{dataloader_idx}",
#                 self._make_rgb_for_tb(y_hat),
#                 global_step=self.current_epoch,
#             )
#             self.logger.experiment.add_images(
#                 f"truth_dl{dataloader_idx}",
#                 self._make_rgb_for_tb(y),
#                 global_step=self.current_epoch,
#             )
#
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         x, y, t = batch
#         y_hat = self.model(x, time=t)
#         loss = (y_hat - y).pow(2).mean()
#         self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
#
#         y_hat_f32, y_f32 = self._to_float_for_metrics(y_hat, y)
#         self.test_metrics(y_hat_f32, y_f32)
#         self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)
#
#         return loss
#
#     def predict_step(self, batch):
#         if len(batch) == 2:
#             x, t = batch
#         else:
#             x, y, t = batch
#         return self.model(x, time=t)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "monitor": "val_loss/dataloader_idx_0",
#                 "interval": "epoch",
#                 "frequency": 1,
#             },
#         }
    # about half as fast as using fp16 00:37<19:39,  1.25it/s

class MultiDataloaderEarlyStopping(Callback):
    """
    Stop training when ALL validation metrics across ALL dataloaders are worsening
    compared to the previous epoch for consecutive epochs.
    """

    def __init__(self, patience=5, verbose=True):
        """
        Args:
            patience: Number of consecutive epochs with all metrics worsening
                     (compared to previous epoch) before stopping
            verbose: If True, print messages about early stopping status
        """
        self.patience = patience
        self.verbose = verbose
        self.wait_count = 0
        self.previous_metrics = {}
        self.metric_directions = {
            'val_loss': 'min',  # Lower is better
            'valid_psnr': 'max',  # Higher is better
            'valid_ssim': 'max',  # Higher is better
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get all logged metrics
        current_metrics = trainer.callback_metrics

        # Extract validation metrics for all dataloaders
        # Format: metric_name/dataloader_idx_N
        val_metrics_by_dl = {}
        for key, value in current_metrics.items():
            if '/dataloader_idx_' in key:
                # Parse metric name and dataloader index
                metric_base, dl_part = key.split('/dataloader_idx_')
                dl_idx = int(dl_part)

                if dl_idx not in val_metrics_by_dl:
                    val_metrics_by_dl[dl_idx] = {}
                val_metrics_by_dl[dl_idx][metric_base] = value.item()

        if not val_metrics_by_dl:
            # No validation metrics found (shouldn't happen)
            return

        # Check if this is the first validation epoch
        if not self.previous_metrics:
            # Initialize - store current as previous for next comparison
            self.previous_metrics = val_metrics_by_dl
            if self.verbose:
                print(f"\nInitialized early stopping tracking with {len(val_metrics_by_dl)} dataloaders")
            return

        # Compare current epoch to previous epoch
        # Check if ALL metrics across ALL dataloaders got worse
        all_worsening = True
        any_metric_found = False

        for dl_idx, current_dl_metrics in val_metrics_by_dl.items():
            if dl_idx not in self.previous_metrics:
                # New dataloader appeared, can't compare
                continue

            for metric_name, current_value in current_dl_metrics.items():
                # Check if we track this metric
                if metric_name not in self.metric_directions:
                    continue

                # Get previous value
                if metric_name not in self.previous_metrics[dl_idx]:
                    continue

                any_metric_found = True
                direction = self.metric_directions[metric_name]
                previous_value = self.previous_metrics[dl_idx][metric_name]

                # Check if this metric got worse compared to previous epoch
                if direction == 'min':
                    got_worse = current_value > previous_value
                else:  # direction == 'max'
                    got_worse = current_value < previous_value

                if not got_worse:
                    # At least one metric improved or stayed same - not all worsening
                    all_worsening = False

        if not any_metric_found:
            # No tracked metrics found for comparison
            self.previous_metrics = val_metrics_by_dl
            return

        # Update wait counter based on whether all metrics worsened
        if all_worsening:
            self.wait_count += 1
            if self.verbose:
                print(f"\nEarly stopping: all metrics worse than previous epoch ({self.wait_count}/{self.patience})")

            if self.wait_count >= self.patience:
                if self.verbose:
                    print(
                        f"\nStopping training: all validation metrics worsened compared to previous epoch for {self.patience} consecutive times")
                trainer.should_stop = True
        else:
            # At least one metric improved or stayed same - reset counter
            if self.wait_count > 0 and self.verbose:
                print(f"\nEarly stopping: reset (at least one metric improved or stayed same)")
            self.wait_count = 0

        # Store current metrics as previous for next comparison
        self.previous_metrics = val_metrics_by_dl

if __name__ == '__main__':
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    input_size = (720 // args.pixel_bin // 8 * 8, 1280 // args.pixel_bin // 8 * 8)
    val_ds = utils.EventImagePairDataset(image_folder=INDIST_IMAGE_PATH,
                                         event_folder=INDIST_EVENT_PATH,
                                         integration_time_s=args.integration_time_s if args.integration_time_s > 0 else 1,
                                         total_time_s=10, start_time_s=5, time_bin=args.time_bin,
                                         pixel_bin=args.pixel_bin,
                                         polarity=args.polarity,
                                         std_channel=args.time_std,
                                         transform=transforms.Compose([np_transforms.CenterCrop(input_size),
                                                                       utils.EventCountNormalization()]))
    _, val_ds, test_ds = utils.data_split(val_ds, validation_split=0.1, testing_split=0.15, seed=47)

    tb_logger = loggers.tensorboard.TensorBoardLogger('lightning_logs', name=args.log_name)
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[ModelCheckpoint(monitor='val_loss', save_top_k=2, save_last=True, mode='min', every_n_epochs=1),
                   LearningRateMonitor(logging_interval='epoch')],
        accelerator='gpu',
        devices=[args.gpu_ind, ],
        max_epochs=args.num_epochs,
    )

    if args.checkpoint_path != '':
        model = Model.load_from_checkpoint(args.checkpoint_path)
        print("loaded from checkpoint: ", args.checkpoint_path)
    else:
        in_channels = args.time_bin * 2 if args.polarity else args.time_bin
        if args.time_std:
            in_channels += 1
        model = Model(dim=64, in_channels=in_channels, lr=args.lr, vanilla_unet=args.vanilla_unet)

        ds = utils.EventImagePairDataset(image_folder=INDIST_IMAGE_PATH,
                                         event_folder=INDIST_EVENT_PATH,
                                         integration_time_s=args.integration_time_s, total_time_s=10, start_time_s=-1,
                                         time_bin=args.time_bin,
                                         pixel_bin=args.pixel_bin,
                                         polarity=args.polarity,
                                         std_channel=args.time_std,
                                         transform=transforms.Compose([
                                             np_transforms.RandomHorizontalFlip(),
                                             np_transforms.RandomVerticalFlip(),
                                             np_transforms.CenterCrop(input_size),
                                             utils.EventCountNormalization()
                                         ]))
        train_ds, _, _ = utils.data_split(ds, validation_split=0.1, testing_split=0.15, seed=47)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True)
        trainer.fit(model, train_dl,
                    DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                               persistent_workers=True))
        print("training finished")

    print("In-distribution testing:")
    trainer.test(model, DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                   persistent_workers=True))

    predictions = trainer.predict(model, dataloaders=DataLoader(test_ds, batch_size=args.batch_size,
                                                                num_workers=args.num_workers, shuffle=False,
                                                                persistent_workers=True))

    ood_ds = utils.EventImagePairDataset(image_folder=OOD_IMAGE_PATH,
                                         event_folder=OOD_EVENT_PATH,
                                         integration_time_s=args.integration_time_s if args.integration_time_s > 0 else 1,
                                         total_time_s=10, start_time_s=5, time_bin=args.time_bin,
                                         pixel_bin=args.pixel_bin,
                                         polarity=args.polarity,
                                         std_channel=args.time_std,
                                         transform=transforms.Compose([np_transforms.CenterCrop(input_size),
                                                                       utils.EventCountNormalization()]),
                                         img_suffix='.png',
                                         calib_img_path=os.path.join(OOD_EVENT_PATH, 'checkerboard0.png'))

    print("Out-of-distribution testing:")
    trainer.test(model, DataLoader(ood_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                   persistent_workers=True))
    predictions_ood = trainer.predict(model, dataloaders=DataLoader(ood_ds, batch_size=args.batch_size,
                                                                    num_workers=args.num_workers, shuffle=False,
                                                                    persistent_workers=True))
    np.savez(os.path.join(tb_logger.log_dir, 'predictions.npz'), pred=np.concatenate(predictions),
             pred_ood=np.concatenate(predictions_ood))
    print("predictions saved to ", os.path.join(tb_logger.log_dir, 'predictions.npz'))
