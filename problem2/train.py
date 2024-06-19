# Course: Computer Vision and Artificial Intelligence for Autonomous Cars, ETH Zurich
# Material for Project 2
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import os
import sys
import argparse
import uuid
from datetime import datetime

import yaml
import shutil
import wandb
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from model import Model
from dataset import DatasetLoader

from utils.task4 import RegressionLoss, ClassificationLoss
from utils.eval import generate_final_predictions, save_detections, generate_submission, compute_map
from utils.vis import point_scene

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dir = self.config['eval']['output_dir']
        if os.path.exists(self.output_dir): shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        self.model = Model(config['model'])
        self.reg_loss = RegressionLoss(config['loss'])
        self.cls_loss = ClassificationLoss(config['loss'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, assinged_target, iou = batch['input'], batch['assinged_target'], batch['iou']
        pred_box, pred_class = self(x)
        loss = self.reg_loss(pred_box, assinged_target, iou) \
               + self.cls_loss(pred_class, iou)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, assinged_target, iou = batch['input'], batch['assinged_target'], batch['iou']
        pred_box, pred_class = self(x)

        loss = self.reg_loss(pred_box, assinged_target, iou) \
               + self.cls_loss(pred_class, iou)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        nms_pred, nms_score = generate_final_predictions(pred_box, pred_class, config['eval'])
        save_detections(os.path.join(self.output_dir, 'pred'), batch['frame'], nms_pred, nms_score)

        # Visualization
        if batch_idx == 0:
            scene = point_scene(batch['points'], nms_pred, batch['target'], name=f'e{self.current_epoch}')
            self.logger.experiment.log(scene, commit=False)

    def validation_epoch_end(self, outputs):
        easy, moderate, hard = compute_map(self.valid_dataset.hf,
                                           os.path.join(self.output_dir, 'pred'),
                                           self.valid_dataset.frames)
        shutil.rmtree(self.output_dir, 'pred')
        self.log('e_map', easy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('m_map', moderate, on_step=False, on_epoch=True, prog_bar=True)
        self.log('h_map', hard, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        frame, x = batch['frame'], batch['input']
        pred_box, pred_class = self(x)
        nms_pred, nms_score = generate_final_predictions(pred_box, pred_class, config['eval'])
        save_detections(os.path.join(self.output_dir, 'test'), frame, nms_pred, nms_score)

    @property
    def submission_file(self):
        return os.path.join(self.output_dir, 'submission.zip')

    def test_epoch_end(self, outputs):
        generate_submission(os.path.join(self.output_dir, 'test'), self.submission_file)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.config['optimizer'])
        scheduler = lrs.MultiStepLR(optimizer, **self.config['scheduler'])
        return [optimizer], [scheduler]

    def setup(self, stage):
        self.train_dataset = DatasetLoader(config=config['data'], split='train')
        self.valid_dataset = DatasetLoader(config=config['data'], split='val')
        self.test_dataset = DatasetLoader(config=config['data'], split='test')

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.config['data']['batch_size'],
                          shuffle=True,
                          pin_memory=True,
                          num_workers=3,
                          collate_fn=self.train_dataset.collate_batch)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=1,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=os.cpu_count(),
                          collate_fn=self.train_dataset.collate_batch)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=os.cpu_count(),
                          collate_fn=self.test_dataset.collate_batch)

def train(config, run_name):

    wandb_logger = WandbLogger(
        name=run_name,
        project='CVAIAC-Ex2',
        save_dir=os.path.join(config["trainer"]["default_root_dir"])
    )

    checkpoint_local_callback = ModelCheckpoint(
        dirpath=os.path.join(config["trainer"]["default_root_dir"], run_name, 'checkpoints'),
    )

    print("Start training", run_name)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=checkpoint_local_callback,
        gpus=-1 if torch.cuda.is_available() else None,
        accelerator='ddp' if torch.cuda.is_available() else None,
        **config['trainer']
    )
    litModel = LitModel(config)
    trainer.fit(litModel)
    trainer.test(litModel)

    wandb.finish()

def adapt_config(config, run_name):
	user = os.getenv('USER') or os.getenv('USERNAME')
	config['data']['root_dir'] = config['data']['root_dir'].replace('$USER', user)
	config['eval']['output_dir'] = config['eval']['output_dir'].replace('$USER', user)
	config['trainer']['default_root_dir'] = config['trainer']['default_root_dir'].replace('$USER', user)
	output_dir = config['eval']['output_dir']
	os.makedirs(output_dir, exist_ok=True)
	run_output_dir = os.path.join(output_dir, run_name)
	os.makedirs(run_output_dir)
	config['eval']['output_dir'] = os.path.join(run_output_dir, 'eval')
	os.makedirs(config['eval']['output_dir'])
	config['trainer']['default_root_dir'] = os.path.join(run_output_dir, 'trainer')
	os.makedirs(config['trainer']['default_root_dir'])
	return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', default='config.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, 'r'))

    timestamp = datetime.now().strftime('%m%d-%H%M')
    run_name = f'G{config["group_id"]}_{timestamp}_{config["name"]}_{str(uuid.uuid4())[:5]}'

    config = adapt_config(config, run_name)    

    train(config, run_name)
