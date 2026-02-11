import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher as CFM
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
import wandb
from tqdm.auto import tqdm

# Trainer class for CFM model training
# NOTE: .sample_location_and_conditional_flow in torchcfm/guided_conditional_flow_matching.py lines 274-...

class Trainer:
    def __init__(
        self,
        model,
        loader,
        val_loader,
        device: str = "cuda",
        batch_size: int = 4,
        epochs: int = 10,
        lr: float = 2e-4,
        loss_type: str = "le",
        scheduler_type: str = None,
        warmup_steps: int = 0,
        lr_min: float = 2e-7,
        gammadecay: float = 0.9999,
        pl_factor: float = 0.5,
        pl_patience: int = 500,
        results_dir: str = "./results_CFM",
        save_every: int = 100,
        wb_run: str = None,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        update_ema_every: int = 100,
        grad_norm: float = 1.0,
        ):
        super().__init__()
        self.model = model
        self.loader = loader
        self.val_loader = val_loader
        self.device = device

        self.batch_size = batch_size
        self.epochs = epochs
        self.step = 0
        
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min
        self.loss_type = loss_type

        self.grad_norm = grad_norm
        
        self.save_every = save_every
        self.results_dir = results_dir
        self.wb_run = wb_run
        
        # EMA model
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.update_ema_every = update_ema_every
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model).eval()
            print("EMA model initialized")
        else:
            self.ema_model = None

        # Optimizer: Adam
        self.opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Learning rate scheduler
        if self.scheduler_type == "cos":
            self.lr_scheduler = CosineAnnealingLR(self.opt, T_max=self.epochs*len(self.loader) - self.warmup_steps, eta_min=self.lr_min)
        elif self.scheduler_type == "exp":
            self.gammadecay = gammadecay
            self.lr_scheduler = ExponentialLR(self.opt, gamma=self.gammadecay)
        elif self.scheduler_type == "plateau":
            self.pl_factor = pl_factor
            self.pl_patience = pl_patience
            self.lr_scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=self.pl_factor, patience=self.pl_patience)
        else:
            self.lr_scheduler = None

        # CFM: exact OT
        self.fm = CFM(sigma=0.0)

        # Initiating validation tracking
        self.val_loss = None
        self.ema_val_loss = None
        self.best_val_loss = float('inf')
        self.best_ema_val_loss = float('inf')

        # Wandb logging
        if self.wb_run is not None:
            self.wandb_log(now=self.wb_run)

    # Compute loss by loss type
    def compute_loss(self, ut, vt):
        """
        Compute loss between reconstructed and original noise at time t
        Args:
        -----
            ut : tensor
                original noise sampled at time t (gauss noise)
            vt : tensor
                reconstructed noise at time t predicted by the model (velocity field)
        Returns:
        --------
            loss : tensor
                computed loss value 
        """
        if self.loss_type == 'l1': # mean absolute error
            loss = torch.mean(torch.abs(ut - vt))
        elif self.loss_type == 'l2': # mean squared error
            loss = F.mse_loss(vt, ut)
        elif self.loss_type == 'le': # weighted sum of l1 and l2 with fixed coefficients
            loss_l1 = (ut - vt).abs().mean()
            loss_l2 = F.mse_loss(vt, ut)
            loss = 0.3 * loss_l1 + 0.7 * loss_l2 # weighted sum coefficients to evaluate
        else: 
            raise NotImplementedError()
        
        return loss

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
    
    def update_ema(self):
        """Update EMA model weights with exponential moving average"""

        if not self.use_ema:
            return
        
        # Update EMA parameters
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1 - self.ema_decay)
    
    def validate(self):
        """Compute validation loss"""

        if self.val_loader is None:
            return None

        self.model.eval()
        val_losses = []

        with torch.inference_mode():
            for batch in self.val_loader:
                batch = self.move_to_device(batch)
                
                image = batch['image']
                mask = batch['mask']
                diagnosis = batch['diagnosis']
                
                # Sample noise and compute validation loss (same as training)
                x0 = torch.randn_like(image)
                t, xt, ut, _, y1 = self.fm.guided_sample_location_and_conditional_flow(x0=x0, x1=image, y1=diagnosis)
                
                vt = self.model(torch.cat([xt, mask], dim=1), t, y1)
                
                loss = self.compute_loss(ut, vt)
                val_losses.append(loss.item())
                
        self.model.train()
        avg_val_loss = np.mean(val_losses)
        return avg_val_loss
    
    def validate_ema(self):
        """Compute validation loss with EMA model"""

        if not self.use_ema or self.ema_model is None or self.val_loader is None:
            return None
        
        self.ema_model.eval()
        val_losses = []
        
        with torch.inference_mode():
            for batch in self.val_loader:
                batch = self.move_to_device(batch)
                
                image = batch['image']
                mask = batch['mask']
                diagnosis = batch['diagnosis']
                
                # Sample noise and compute validation loss with EMA model
                x0 = torch.randn_like(image)
                t, xt, ut, _, y1 = self.fm.guided_sample_location_and_conditional_flow(x0=x0, x1=image, y1=diagnosis)
                
                vt = self.ema_model(torch.cat([xt, mask], dim=1), t, y1)
                
                loss = self.compute_loss(ut, vt)
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def save_checkpoint(self, avg_loss=None):
        """
        Save model checkpoint
        Args:
        -----
            save_dir: directory
                where to save checkpoint
            avg_loss: float
                current average loss value
        """
        checkpoint = {
            'step': self.step,
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'ema_model': self.ema_model.state_dict() if self.ema_model is not None else None,
            'val_loss': self.val_loss,
            'avg_loss': avg_loss,
            'ema_val_loss': self.ema_val_loss,
        }

        path = f"{self.results_dir}/checkpoint_{self.step}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint {self.step} saved: {path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state
        Args:
        -----
            checkpoint_path: str
                path to checkpoint file
        Returns:
        --------
            step: int
                training step to resume from
        Note:
        -----
            This function assumes that the checkpoint contains the model state dict,
            optimizer state dict, and optionally the lr_scheduler and ema_model state dicts.        
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.opt.load_state_dict(checkpoint['optimizer'])
        
        if self.lr_scheduler is not None and checkpoint.get('lr_scheduler'):
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        if self.use_ema and checkpoint.get('ema_model'):
            self.ema_model.load_state_dict(checkpoint['ema_model'])
        
        self.step = checkpoint.get('step', 0)
        print(f"Resumed from step {self.step}")
        return self.step
    
    def wandb_log(self, now):
        """Initialize wandb logging"""
        
        #log with weights and biases 
        batch_size = self.loader.batch_size if hasattr(self.loader, 'batch_size') else 'unknown'
        
        wandb.init(project="med-cfm", name=f"{now}",
                   config={
                        "epochs": self.epochs,
                        "batch_size": batch_size,
                        "learning_rate": self.lr,
                        "lr_warmup_steps": self.warmup_steps,
                        "lr_min": self.lr_min,
                        "lr_scheduler": self.scheduler_type,
                        "loss_type": self.loss_type,
                        "save_every": self.save_every,
                        "use_ema": self.use_ema,
                        "ema_decay": self.ema_decay,
                        "grad_norm": self.grad_norm
                        }
                )
    

    def train(self):
        """
        Main training loop over epochs = total steps
        """
        
        os.makedirs(self.results_dir, exist_ok=True)

        pbar = tqdm(total=self.epochs*len(self.loader), desc="Training", unit="step")
        
        self.model.train()
         
        checkpoint_losses = [] # windowed: reset after each checkpoint

        for epoch in range(self.epochs):
            for i, batch in enumerate(self.loader):

                step=epoch*len(self.loader) + i 
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
                    y1  # from .sample_location_and_conditional_flow
                )

                # Compute MSE loss between predicted and target velocity
                #loss = torch.mean((vt - ut) ** 2)
                loss = self.compute_loss(ut, vt)

                # Optimization step
                self.opt.zero_grad()
                loss.backward()
                # gradient clipping
                if self.grad_norm is not None:
                    clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
                self.opt.step()
                
                # Update EMA model
                if self.step % self.update_ema_every == 0:
                    self.update_ema()

                # linear warmup and lr scheduling if/which applied
                if self.step < self.warmup_steps:
                    lr_scale = (self.step + 1) / self.warmup_steps
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = self.lr * lr_scale

                elif self.step == self.warmup_steps:
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = self.lr
                    print(f"Warmup complete at step {step}")

                elif step > self.warmup_steps and self.lr_scheduler is not None:
                    if not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                        self.lr_scheduler.step()

                # Track lr, loss
                current_lr = self.opt.param_groups[0]['lr']
                current_loss = loss.item()
                checkpoint_losses.append(current_loss)

                pbar.set_postfix({
                    "loss": current_loss,
                    "lr": current_lr,
                    "ema_val_loss": self.ema_val_loss if self.ema_val_loss is not None else -1
                })
                pbar.update(1)


                # Log training info to wandb
                if self.wb_run is not None:
                    wandb.log({"step": step,
                               "learning_rate": current_lr,
                               "training_loss": current_loss,
                               #"ema_loss": self.ema_val_loss
                    })
                    
                # Save checkpoint #
                if step % self.save_every == 0:
                    # Average loss over checkpoint window
                    avg_loss = np.mean(checkpoint_losses)
                    checkpoint_losses = [] 

                    # Validation loss: if validation set provided
                    if self.val_loader is not None:
                        self.val_loss = self.validate()
                        if self.use_ema:
                            self.ema_val_loss = self.validate_ema()

                        # Update best validation loss
                        if self.val_loss is not None and self.val_loss < self.best_val_loss:
                            self.best_val_loss = self.val_loss
                        if self.ema_val_loss is not None and self.ema_val_loss < self.best_ema_val_loss:
                            self.best_ema_val_loss = self.ema_val_loss
                        
                        # Step ReduceLROnPlateau with validation loss
                        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                            self.lr_scheduler.step(self.val_loss)
                                    
                    # Checkpoint saving
                    self.save_checkpoint(avg_loss=avg_loss)
                    print(f"Step [{step}/{self.epochs*len(self.loader)}] - Checkpoint saved\n")

                    if self.wb_run is not None:
                        wandb.log({"val_loss": self.val_loss if self.val_loss is not None else -1,
                                   "ema_val_loss": self.ema_val_loss if self.ema_val_loss is not None else -1,
                                   "best_val_loss": self.best_val_loss if self.val_loss is not None else -1,
                                   "best_ema_val_loss": self.best_ema_val_loss if self.ema_val_loss is not None else -1,
                                   "avg_loss": avg_loss
                        })
        pbar.close()
                    
