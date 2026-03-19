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

# NOTE: careful with loss type see Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport
# see eq. 10 and th. 3.2 is it okay to alse use abs diff? and consequently le also?
# looks like not: mse baset ot , loss during training le should be best bus see l1 smooth and t dependence commented out

# NOTE: no gradient accumalation instead of per image in batch for greater numerical stability and it wouldn't change much


def perturb_mask(mask, apply_prob=0.7, p_morph=0.25, p_dropout=0.02, noise_std=0.05):
    """
    Mask modification (for training only) : #boundary perturbation (dilation/erosion), gaussian noise, spatial dropout and reinstatement to original range
    Args:
    -----   
        mask: tensor
            input mask to perturb
        apply_prob: float
            probability to apply augmentation pipeline
        p_morph: float
            probability to apply morphological perturbation (dilation/erosion)
        p_dropout: float
            probability for spatial dropout
        noise_std: float
            standard deviation for gaussian noise
    Returns:
    --------       
        perturbed_mask: tensor
            augmented mask with same shape as input = noisy and slightly morphologically perturbed version
    Note:
    -----
        To void operations set all values to 0
    """

    # Track original range (any range, not only [-1,1]) #keepdim=True to keep dimension for final range reinstatement)
    mask_min = mask.amin(dim=(2, 3, 4), keepdim=True) # find minimum value in values in dimensions (D,H,W)=(2,3,4) 
    mask_max = mask.amax(dim=(2, 3, 4), keepdim=True) # find maximum value in values in dimensions (D,H,W)=(2,3,4)

    # Apply or skip entirely: some noisy some not to not overfit over noisy masks and the clean ones' benefits
    if torch.rand(1, device=mask.device) > apply_prob:
        return mask

    # Too much according to Tom, gauss noise already modifies edges a bit?
    ## 1. Boundary perturbation with dilation/erosio = morphological variation over random kernel (3-5)
    #if torch.rand(1, device=mask.device) < p_morph:
    #    k = torch.randint(1, 3, (1,), device=mask.device).item() * 2 + 1  # kernel = 3 or 5
    #    if torch.rand(1, device=mask.device) < 0.5: # dilation
    #        mask = F.max_pool3d(mask, kernel_size=k, stride=1, padding=k//2)
    #    else: # erosion
    #        mask = -F.max_pool3d(-mask, kernel_size=k, stride=1, padding=k//2)

    # 2. Gaussian noise over strict values to smooth vt?
    if noise_std > 0:
        mask = mask + noise_std * torch.randn_like(mask) # x_gn = x + sigma*eps , eps~N(0,1) 

    # 3. Spatial dropout to help with generation with incomplete masks!!
    if p_dropout > 0:
        keep = (torch.rand_like(mask) > p_dropout).float()
        mask = mask * keep + mask_min * (1.0 - keep) # dropped voxels go to background-like value = minimum in current mask

    # Reinstate to original range
    mask = torch.max(torch.min(mask, mask_max), mask_min)

    return mask

class Trainer:
    def __init__(
        self,
        model,
        loader,
        val_loader,
        device: str = "cuda",
        batch_size: int = 4,
        epochs: int = 1000,
        lr: float = 2e-4,
        loss_type: str = "le",
        scheduler_type: str = None,
        warmup_steps: int = 0,
        lr_min: float = 2e-7,
        gamma_decay: float = 0.9999,
        #pl_factor: float = 0.5,
        #pl_patience: int = 500,
        results_dir: str = "./results_CFM",
        save_every: int = 100,
        wb_run: str = None,
        use_ema: bool = True,
        ema_decay: float = 0.995,
        update_ema_every: int = 100,
        grad_norm: float = 1.0,
        weight_decay: float = 1e-4,
        val_seeds: list = [42, 135, 654]
        ):
        super().__init__()
        self.model = model
        self.loader = loader
        self.val_loader = val_loader
        self.device = device

        self.batch_size = batch_size
        self.epochs = epochs
        self.step = 0
        self.tot_steps = self.epochs * len(self.loader) * self.batch_size #able to write like this due to drop_last = True in dataloader, making len(loader) constant
        
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min
        self.loss_type = loss_type

        self.grad_norm = grad_norm
        self.weight_decay = weight_decay

        self.val_seeds = val_seeds
        
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
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Learning rate scheduler
        if self.scheduler_type == "cos":
            self.lr_scheduler = CosineAnnealingLR(self.opt, T_max=self.tot_steps - self.warmup_steps, eta_min=self.lr_min)
        elif self.scheduler_type == "exp":
            self.gamma_decay = gamma_decay
            self.lr_scheduler = ExponentialLR(self.opt, gamma=self.gamma_decay)
        #elif self.scheduler_type == "plateau":
        #    self.pl_factor = pl_factor
        #    self.pl_patience = pl_patience
        #    self.lr_scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=self.pl_factor, patience=self.pl_patience)
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
            loss = torch.mean(torch.abs(ut - vt)) # variante smooth/huber loss : F.smooth_l1_loss(vt, ut, beta=0.05)
        elif self.loss_type == 'l2': # mean squared error
            loss = F.mse_loss(vt, ut)
        elif self.loss_type == 'le': # weighted sum of l1 and l2 with fixed coefficients
            loss_l1 = torch.mean(torch.abs(ut - vt)) # variante smooth/huber loss : F.smooth_l1_loss(vt, ut, beta=0.05)
            loss_l2 = F.mse_loss(vt, ut)
            loss = 0.3 * loss_l1 + 0.7 * loss_l2 # weighted sum coefficients to evaluate
            # lambda_t = 0.05 * t**2 
            # loss = mse + lambda_t * smooth_l1

        else: 
            raise NotImplementedError()
        
        return loss
    
    def update_ema(self):
        """Update EMA model weights with exponential moving average
            ema_p_new = ema_decay * ema_p_old + (alpha = 1 - ema_decay) * model_p_current"""
        
        ## Move EMA model to CPU for memory efficiency during update
        #self.ema_model.cpu()

        # Update EMA parameters
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()): # paired ema and model parameters
                ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1 - self.ema_decay)
    
    @torch.inference_mode() # no grad + eval for layers like dropout, batchnorm
    def validate(self):
        """Compute validation loss with multiple fixed seeds for reproducibility and reduced bias"""

        self.model.eval()

        seed_val_losses = []  # Store average loss for each seed

        # Run validation with multiple seeds
        for seed in self.val_seeds:
            # Set seed for reproducible validation
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            for batch in self.val_loader:
                #batch = self.move_to_device(batch)
                
                image = batch['image']
                mask = batch['mask']
                diagnosis = batch['diagnosis']
                
                # Sample noise and compute validation loss
                x0 = torch.randn_like(image)

                # Keep mask and diagnosis aligned with OT permutation by packing both into y1
                B = image.shape[0]
                diagnosis_scalar = diagnosis.view(B, -1)[:, :1]
                diagnosis_map = diagnosis_scalar.to(mask.dtype).view(B, 1, 1, 1, 1).expand(-1, 1, *mask.shape[2:])
                cond = torch.cat([mask, diagnosis_map], dim=1)

                t, xt, ut, _, cond_ot = self.fm.guided_sample_location_and_conditional_flow(x0=x0, x1=image, y1=cond)
                mask_ot = cond_ot[:, :1]
                diagnosis_ot = cond_ot[:, 1, 0, 0, 0].to(diagnosis.dtype)
                
                for t_i, xt_i, ut_i, y1_i , mask_i in zip(t, xt, ut, diagnosis_ot, mask_ot): 
                    t_i = t_i.unsqueeze(0).to(self.device)#, non_blocking = True)
                    xt_i = xt_i.unsqueeze(0).to(self.device)#, non_blocking = True)
                    ut_i = ut_i.unsqueeze(0).to(self.device)#, non_blocking = True)
                    y1_i = y1_i.unsqueeze(0).to(self.device)#, non_blocking = True) 
                    mask_i = mask_i.unsqueeze(0).to(self.device)#, non_blocking = True)

                    vt = self.model(torch.cat([xt_i, mask_i], dim=1), t_i, y1_i )

                    loss = self.compute_loss(ut_i, vt)
                    seed_val_losses.append(loss.item())

                    # Explicit tensor cleanup for memory efficiency
                    del t_i, xt_i, ut_i, y1_i, mask_i, vt, loss
        
        # Store average loss for this seed
        avg_seed_loss = np.mean(seed_val_losses)

        # Explicitly clear CUDA cache after validation to free up memory for training
        del t, xt, ut, cond_ot, mask_ot, diagnosis_ot, image, mask, diagnosis
        torch.cuda.empty_cache()
       
        # Return mean across all seeds for more robust validation metric
        return avg_seed_loss
    
    @torch.inference_mode()
    def validate_ema(self):
        """Compute validation loss with EMA model using multiple fixed seeds for reproducibility"""
        self.ema_model.eval()

        seed_val_losses = []
        
        # Run validation with multiple seeds
        for seed in self.val_seeds:
            # Set seed for reproducible validation
            torch.manual_seed(seed)
            np.random.seed(seed)

            for batch in self.val_loader:
                
                image = batch['image']
                mask = batch['mask']
                diagnosis = batch['diagnosis']
                
                # Sample noise and compute validation loss with EMA model
                x0 = torch.randn_like(image)
                # Keep mask and diagnosis aligned with OT permutation by packing both into y1
                B = image.shape[0]
                diagnosis_scalar = diagnosis.view(B, -1)[:, :1]
                diagnosis_map = diagnosis_scalar.to(mask.dtype).view(B, 1, 1, 1, 1).expand(-1, 1, *mask.shape[2:])
                cond = torch.cat([mask, diagnosis_map], dim=1)

                t, xt, ut, _, cond_ot = self.fm.guided_sample_location_and_conditional_flow(x0=x0, x1=image, y1=cond)
                mask_ot = cond_ot[:, :1]
                diagnosis_ot = cond_ot[:, 1, 0, 0, 0].to(diagnosis.dtype)
                
                for t_i, xt_i, ut_i, y1_i , mask_i in zip(t, xt, ut, diagnosis_ot, mask_ot): 
                    t_i = t_i.unsqueeze(0).to(self.device)#, non_blocking = True)
                    xt_i = xt_i.unsqueeze(0).to(self.device)#, non_blocking = True)
                    ut_i = ut_i.unsqueeze(0).to(self.device)#, non_blocking = True)
                    y1_i = y1_i.unsqueeze(0).to(self.device)#, non_blocking = True) 
                    mask_i = mask_i.unsqueeze(0).to(self.device)#, non_blocking = True)

                    vt = self.ema_model(torch.cat([xt_i, mask_i], dim=1), t_i, y1_i )
                    
                    loss = self.compute_loss(ut_i, vt)
                    seed_val_losses.append(loss.item())

                    # Explicit tensor cleanup for memory efficiency
                    del t_i, xt_i, ut_i, y1_i, mask_i, vt, loss
            
        # Store average loss for this seed
        avg_seed_loss = np.mean(seed_val_losses)

        # Explicitly clear CUDA cache after validation to free up memory for training
        del t, xt, ut, cond_ot, mask_ot, diagnosis_ot, image, mask, diagnosis
        torch.cuda.empty_cache()
    
        # Return mean across all seeds for more robust validation metric
        return avg_seed_loss
    
    
    def save_checkpoint(self, avg_loss=None):
        """
        Save model checkpoint
        Args:
        -----
            avg_loss: float
                current average loss value
        """

        def to_cpu(obj):
            """Recursively move tensors in nested dicts/lists to CPU"""

            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().clone()  # .detach() per sicurezza extra
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(to_cpu(v) for v in obj)
            else:
                return obj


        checkpoint = {
            'step': self.step,
            'model': to_cpu(self.model.state_dict()),
            'optimizer': to_cpu(self.opt.state_dict()),
            'lr_scheduler': to_cpu(self.lr_scheduler.state_dict()) if self.lr_scheduler else None,
            'ema_model': to_cpu(self.ema_model.state_dict()) if (self.use_ema and self.ema_model) else None,
            'val_loss': self.val_loss,
            'avg_loss': avg_loss,
            'ema_val_loss': self.ema_val_loss,
        }

        path = (f"{self.results_dir}/checkpoint_{self.step//self.save_every}.pt" if self.step != self.tot_steps - 1 else f"{self.results_dir}/checkpoint_final.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint {self.step} saved: {path}")

        # Explicit cleanup to free memory after saving checkpoint
        del checkpoint
        torch.cuda.empty_cache()

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
                        "total_steps": self.tot_steps,
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

        pbar = tqdm(total=self.tot_steps, desc="Training", unit="step")
        
        self.model.train() # set model to training mode for layers like dropout, batchnorm
         
        checkpoint_losses = [] # windowed: reset after each checkpoint

        for epoch in range(self.epochs):
            for i, batch in enumerate(self.loader):

                image = batch["image"]        # (B, 1, D, H, W)
                mask = batch["mask"]          # (B, 1, D, H, W)
                diagnosis = batch["diagnosis"]  # (B,)

                # Sample noise as source
                x0 = torch.randn_like(image)

                # Sample time, location, and conditional flow
                # Keep mask and diagnosis aligned with OT permutation by packing both into y1
                B = image.shape[0] # batch size
                diagnosis_scalar = diagnosis.view(B, -1)[:, :1] # batch size, 1 --> scalar diagnosis
                diagnosis_map = diagnosis_scalar.to(mask.dtype).view(B, 1, 1, 1, 1).expand(-1, 1, *mask.shape[2:]) # batch size, 1, D, H, W --> match mask domension for concatenation
                cond = torch.cat([mask, diagnosis_map], dim=1) # concatenate mask and diagnosis to keep track during OT permutation

                t, xt, ut, _, cond_ot = self.fm.guided_sample_location_and_conditional_flow(x0=x0, x1=image, y1=cond)
                mask_ot = cond_ot[:, :1] # OT permuted mask
                diagnosis_ot = cond_ot[:, 1, 0, 0, 0].to(diagnosis.dtype) # OT permuted diagnosis scalar

                # mask augmentation AFTER OT (so that OT image-mask pairing stays consistent?)
                mask_ot_noisy = perturb_mask( mask_ot, apply_prob=0.7, p_morph=0.25, p_dropout=0.02, noise_std=0.05)

                #move to GPU if available: single img possible bc order manteined, add back batch dimension
                for t_i, xt_i, ut_i, y1_i , mask_i in zip(t, xt, ut, diagnosis_ot, mask_ot_noisy): 
                    t_i = t_i.unsqueeze(0).to(self.device, non_blocking = True)
                    xt_i = xt_i.unsqueeze(0).to(self.device, non_blocking = True)
                    ut_i = ut_i.unsqueeze(0).to(self.device, non_blocking = True)
                    y1_i = y1_i.unsqueeze(0).to(self.device, non_blocking = True) 
                    mask_i = mask_i.unsqueeze(0).to(self.device, non_blocking = True)

                    # Model prediction: concatenate xt with mask, pass time and diagnosis
                    vt_i = self.model( # vt=predicted velocity
                        torch.cat([xt_i, mask_i], dim=1),  # Concatenate along channel dimension to condition on mask
                        t_i, # time
                        y1_i  # from .sample_location_and_conditional_flow
                    )

                    # Compute MSE loss between predicted and target velocity
                    loss = self.compute_loss(ut_i, vt_i)

                    # Backpropagation and optimization ##only backprop for cmulative alternative
                    self.opt.zero_grad() 
                    loss.backward()

                    # gradient clipping
                    clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)

                    # Optimization step    
                    self.opt.step()
                
                    # Update EMA model
                    if self.step % self.update_ema_every == 0:
                        if self.use_ema:
                            self.update_ema()

                    # linear warmup and lr scheduling if/which applied
                    if self.step < self.warmup_steps:
                        lr_scale = (self.step + 1) / self.warmup_steps
                        for param_group in self.opt.param_groups:
                            param_group['lr'] = self.lr * lr_scale

                    elif self.step == self.warmup_steps:
                        for param_group in self.opt.param_groups:
                            param_group['lr'] = self.lr
                        print(f"Warmup complete at step {self.step}")

                    elif self.step > self.warmup_steps and self.lr_scheduler is not None:
                        #if not isinstance(self.lr_scheduler, ReduceLROnPlateau):
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
                        wandb.log({"step": self.step,
                                   "learning_rate": current_lr,
                                   "training_loss": current_loss,
                                   #"ema_loss": self.ema_val_loss
                        })

                    # Save checkpoint #
                    if self.step % self.save_every == 0 and self.step != 0 or self.step == self.tot_steps - 1:
                        with torch.no_grad():

                            # Average loss over checkpoint window
                            avg_loss = np.mean(checkpoint_losses)
                            checkpoint_losses = [] 

                            # Validation loss: if validation set provided
                            if self.val_loader is not None:
                                self.val_loss = self.validate()

                                ## Step ReduceLROnPlateau with validation loss
                                #if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                                #    self.lr_scheduler.step(self.val_loss)

                            if self.use_ema and self.val_loader is not None:
                                self.ema_val_loss = self.validate_ema()

                            # Update best validation loss
                            if self.val_loss is not None and self.val_loss < self.best_val_loss:
                                self.best_val_loss = self.val_loss

                            if self.ema_val_loss is not None and self.ema_val_loss < self.best_ema_val_loss:
                                self.best_ema_val_loss = self.ema_val_loss

                            # Checkpoint saving
                            self.save_checkpoint(avg_loss=avg_loss)
                            print(f"Step [{self.step}/{self.tot_steps}] - Checkpoint saved\n")

                            if self.wb_run is not None: 
                                wandb.log({"val_loss": self.val_loss,# values can be None, run won't fail
                                           "ema_val_loss": self.ema_val_loss,
                                           "best_val_loss": self.best_val_loss,
                                           "best_ema_val_loss": self.best_ema_val_loss,
                                           "avg_loss": avg_loss
                                })
                    
                    # Increment step counter
                    self.step += 1

                    # Explicit cleanup to free memory after each step
                    #torch.cuda.synchronize()
                    #torch.cuda.empty_cache()

                    self.model.train() # set model to training mode for layers like dropout, batchnorm after validation

        pbar.close()

