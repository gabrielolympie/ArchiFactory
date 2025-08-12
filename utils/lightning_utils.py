from bitsandbytes.optim.ademamix import AdEMAMix
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
import gc
import os

torch.set_float32_matmul_precision('medium')
        
class LightningWrapper(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=2e-4,
        betas=(0.9, 0.999, 0.9999),
        alpha=5,
        weight_decay=1e-4,
        warmup_steps=50,
        total_steps=10000,
        min_lr=1e-6
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.betas = betas
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, input_ids, return_loss=False):
        return self.model(input_ids, return_loss=return_loss)
    
    def training_step(self, batch):
        
        loss=self.model(batch, return_loss=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch):
        loss=self.model(batch, return_loss=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdEMAMix(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            alpha=self.alpha,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.total_steps,  # Adjusted total steps
            eta_min=self.min_lr  # Minimum learning rate
        )
        
        if self.warmup_steps > 0:
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))
                return scheduler.get_lr()[0] / self.learning_rate
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return [optimizer], [{"scheduler": warmup_scheduler, "interval": "step"},{"scheduler": scheduler, "interval": "step", 'start_epoch':self.warmup_steps}]


        return {
           "optimizer": optimizer,
           "lr_scheduler": {"scheduler": scheduler,"interval": "step"}
        }
        
class LightningDataset(Dataset):
    def __init__(self, data, max_length):
        self.data = data
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        return text
        
def prepare_model_and_data(
    model,
    train_data,
    val_data,
    epochs,
    batch_size,
    max_length,
    learning_rate=2e-4,
    betas=(0.9, 0.999, 0.9999),
    alpha=5,
    weight_decay=1e-4,
    warmup_steps=64,
    min_lr=1e-7,
    gradient_accumulation_steps=None,
    log_dir='runs',
    log_name='train',
    checkpoint_dir='checkpoints',
    checkpoint_every_n_steps=200,
):
    total_steps = len(train_data) * epochs
    
    ## Create lightning model
    lightning_model = LightningWrapper(
        model,
        learning_rate=learning_rate,
        betas=betas,
        alpha=alpha,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr
    )
    
    ## Create checkpoint dir and callback
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=log_name + "-{epoch}-{step}-{train_loss:.4f}",
        save_on_train_epoch_end=False,  # Allow saving during training steps
        every_n_train_steps=checkpoint_every_n_steps,  # Now saves every n steps
        save_top_k=3,  # Save all checkpoints (-1 means keep all)
        monitor="train_loss",
        save_last=True  # Save the last checkpoint
    )
    
    ## Create logger
    logger = TensorBoardLogger(log_dir, name=log_name)
    
    ## Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=[checkpoint_callback],
        logger=logger,
        accumulate_grad_batches=gradient_accumulation_steps,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=1,
    )
    
    ## Create dataset
    train_dataset = LightningDataset(train_data, max_length)
    val_dataset = LightningDataset(val_data, max_length)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    
    return lightning_model, trainer, train_loader, val_loader


