
import os
import numpy as np
from tomlkit import datetime
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher as CFM
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
import wandb

# Trainer class for CFM model training
# TODO: implement validation, ema model 
# NOTE: .sample_location_and_conditional_flow in torchcfm/guided_conditional_flow_matching.py lines 274-...

class Trainer:
    def __init__(
        self,
        model,
        loader,
        val_loader,
        device,
        lr = 2e-4,
        epochs = 1000,
        loss_type = "le",
        scheduler_type = None,
        warmup_steps = 0,
        lr_min = 2e-7,
        results_dir = "./results_CFM",
        save_every = 100,
        wb=False,
        ):
        super().__init__()
        self.model = model
        self.loader = loader
        self.val_loader = val_loader
        self.device = device
        self.step = 0
        self.epochs = epochs
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.results_dir = results_dir
        self.save_every = save_every
        self.loss_type = loss_type
        self.lr_min = lr_min
        self.wb = wb

        # Optimizer: Adam
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)

        # CFM instantiation
        self.fm = CFM(sigma=0.0)

    # Compute loss from loss type
    def compute_loss(self, noise, x_recon):
        """
        Compute loss between reconstructed and original noise
        Args:
        -----
            x_recon : tensor
            noise : tensor
        """
        if self.loss_type == 'l1':
            #loss = (noise - x_recon).abs().mean()
            loss = torch.mean(torch.abs(noise - x_recon))
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == 'le':
            loss_l1 = (noise - x_recon).abs().mean()
            loss_l2 = F.mse_loss(x_recon, noise)
            loss = 0.3 * loss_l1 + 0.7 * loss_l2 # weighted sum coffincients to evaluate
        else:
            raise NotImplementedError()
        
        return loss
    
    def schedule_lr(self):
        """
        Learning rate scheduler step
        """
        if self.step < self.warmup_steps:
            lr_scale = float(self.step + 1) / float(self.warmup_steps)
            current_lr = self.lr * lr_scale
            for param_group in self.opt.param_groups:
                param_group['lr'] = current_lr
        elif self.step == self.warmup_steps: # at the end of warmup, set lr to train_lr if not already reached
            for param_group in self.opt.param_groups:
                param_group['lr'] = self.lr
                print(f"LR warmup completed, current LR set: {current_lr}, step: {self.step}")
                print(f"Starting LR scheduler: CosineAnnealingLR with min LR: {self.lr_min}")
        else:    
            if self.scheduler_type == "cos":
                self.lr_scheduler = CosineAnnealingLR(self.opt, T_max=self.epochs - self.warmup_steps, eta_min=self.lr_min)
            elif self.scheduler_type == "exp":
                self.lr_scheduler = ExponentialLR(self.opt, gamma=0.9999)
            elif self.scheduler_type == "plateau":
                self.lr_scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=500, verbose=True)
            elif self.scheduler_type is None:
                pass
            else:
                raise NotImplementedError(f"LR scheduler {self.scheduler_type} not implemented")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def move_to_device(self, batch):
        """
        Move all tensors in the batch to the specified device (CPU/GPU)
        Args:
        -----
            batch : dictionary
                dictionary containing tensors = values
        Returns:
        --------
            dict : dictionary
                dictionary with tensors moved to the target device 
        """
        return {
            k: v.to(self.device, non_blocking=True) 
            for k, v in batch.items()
        }
    
    def validate(self):
        val_loss=None
        return val_loss
    
    def validate_ema(self):
        ema_val_loss=None
        return ema_val_loss
    
    def save_checkpoint(self, avg_loss=None, val_loss=None, ema_val_loss=None):
        """
        Save model checkpoint
        Args:
        -----
            save_dir: directory
                where to save checkpoint
            epoch: int
                current epoch number
            loss: float
                current loss value
        """
        checkpoint = {
            'step': self.step,
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'val_loss': val_loss,
            'avg_loss':avg_loss,
            'ema_val_loss': ema_val_loss,
            #ema
            #best_val_loss
            #best_milestone
            #val_losses
        }

        path = f"{self.results_dir}/checkpoint_{self.step}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint {self.step} saved: {path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        Args:
        -----
            checkpoint_path: path to checkpoint file        
        Returns:
        --------
            checkpoint epoch number
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.opt.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint.get('step', 0)
        return checkpoint['epoch']
    
    def wandb_log(self):
        if self.wb:
            #log with weights and biases 
            now=self.wb
            wandb.init(project="med-ddpm", name=f"{now} ",
                       config={
                            "epochs": self.epochs,
                            "learning_rate": self.lr,
                            "lr_warmup_steps": self.warmup_steps,
                            "lr_min": self.lr_min,
                            "batch_size" : self.batch_size,
                            "steps" : self.step,
                            "update_ema_every" : self.update_ema_every,
                            "gradient_accumulate_every": self.gradient_accumulate_every,
                            "sample_every" : self.save_and_sample_every,
                            "initial_weights" :self.initial_weights
                            }
                    )
    

    def train(self):
        """
        Main training loop over epochs = total steps
        """
        
        os.makedirs(self.results_dir, exist_ok=True)
        self.model.train()
         
        running_losses = [] # to track losses for logging    

        while self.step < self.epochs:
            for batch in self.loader:
                batch = self.move_to_device(batch)

                image = batch["image"]        # (B, 1, D, H, W)
                mask = batch["mask"]          # (B, 1, D, H, W)
                diagnosis = batch["diagnosis"]  # (B,)

                # Sample noise as source
                x0 = torch.randn_like(image)

                # Sample time, location, and conditional flow
                t, xt, ut, _, y1= self.fm.guided_sample_location_and_conditional_flow(x0=x0, x1=image, y1=diagnosis)

                # Model prediction: concatenate xt with mask, pass time and diagnosis
                vt = self.model( # vt=predicted velocity
                    torch.cat([xt, mask], dim=1),  # Concatenate along channel dimension to condition on mask
                    t, # time
                    y1  # exit from .sample_location_and_conditional_flow
                )

                # Compute MSE loss between predicted and target velocity
                #loss = torch.mean((vt - ut) ** 2)
                loss = self.compute_loss(ut, vt)

                # Optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # lr scheduler step by calling schedule_lr function
                self.schedule_lr()
                current_lr = self.opt.param_groups[0]['lr']
                
                self.step += 1
                
                # Track loss
                current_loss = loss.item()
                running_losses.append(current_loss) # append current loss to list
                avg_loss = np.mean(running_losses)

                # Print
                print(f"Step [{self.step}/{self.epochs}] - Loss: {current_loss:.4f}")
                
                # Log training info to weights and biases
                wandb.log({"step": self.step,
                           "learning_rate": current_lr,
                           "training_loss": avg_loss
                })

                # Save checkpoint
                if self.step % self.save_every == 0:
                    # Overall average loss, validation loss, ema validation loss
                    if self.val_loader is not None:
                        val_loss=self.validate()
                        ema_val_loss=self.validate_ema()

                        wandb.log({"val_loss": val_loss,
                                   #"best_val_loss": self.best_val_loss,
                                   #"best_ema_val_loss": self.best_ema_val_loss,
                                   "ema_val_loss": ema_val_loss
                        })
                    else:
                        val_loss = None
                        ema_val_loss = None

                    # Checkpoint saving
                    self.save_checkpoint(val_loss=val_loss, avg_loss=avg_loss, ema_val_loss=ema_val_loss)
                    print(f"Step [{self.step}/{self.epochs}] - Checkpoint saved\n")

                    # Sampling and visualization can be added here (see ddpm trainer)

                    
                
    
    
