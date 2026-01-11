import copy
import functools
import os
import glob

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.utils.tensorboard
from torch.optim import AdamW
import torch.cuda.amp as amp
import wandb

import itertools
import numpy as np

from . import dist_util, logger
from .resample import LossAwareSampler, UniformSampler
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    if _max > _min:
        normalized_img = (img - _min) / (_max - _min)
    else:
        normalized_img = np.zeros_like(img)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        val_data=None,
        batch_size,
        in_channels,
        image_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        contr,
        save_interval,
        resume_checkpoint,
        resume_step,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dataset='brats',
        summary_writer=None,
        mode='default',
        loss_level='image',
        sample_schedule='direct',
        diffusion_steps=1000,
        wavelet='haar',
        special_checkpoint_steps=None,
        save_to_wandb=True,
        val_interval=1000
    ):
        self.summary_writer = summary_writer
        self.mode = mode
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.val_datal = val_data
        self.val_interval = val_interval
        self.dataset = dataset
        self.iterdatal = iter(data)
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.contr = contr
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.grad_scaler = amp.GradScaler()
        else:
            self.grad_scaler = amp.GradScaler(enabled=False)
        
        # Initialize schedule_sampler (will be None for DirectRegressionLoop)
        if schedule_sampler is not None:
            self.schedule_sampler = schedule_sampler
        elif hasattr(self, '_skip_schedule_sampler'):
            # DirectRegressionLoop sets this flag before calling super().__init__
            self.schedule_sampler = None
        else:
            # Default: create UniformSampler for regular diffusion training
            self.schedule_sampler = UniformSampler(diffusion, maxt=diffusion.num_timesteps)
        
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        
        # Set wavelet parameters BEFORE initializing DWT/IDWT
        self.wavelet = wavelet
        self.loss_level = loss_level
        self.sample_schedule = sample_schedule
        self.diffusion_steps = diffusion_steps
        
        # Special checkpoint configuration
        self.special_checkpoint_steps = special_checkpoint_steps
        self.save_to_wandb = save_to_wandb
        self.saved_special_checkpoints = set()  # Track which special checkpoints we've saved
        
        # Initialize wavelet transforms (requires self.wavelet to be set)
        # Only create DWT/IDWT if using wavelets (not None or "null")
        if self.wavelet and self.wavelet != 'null':
            self.dwt = DWT_3D(self.wavelet)
            self.idwt = IDWT_3D(self.wavelet)
        else:
            self.dwt = None
            self.idwt = None
        
        # ✅ FIXED: Proper step counter initialization
        # Start from resume_step if resuming, otherwise start from 0
        self.step = resume_step if resume_step else 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.sync_cuda = th.cuda.is_available()
        
        # Track best MSE loss (lower is better)
        self.best_losses = {}
        self.best_checkpoints = {}
        self.best_val_loss = float('inf')
        self.best_val_checkpoint = None
        self.checkpoint_dir = os.path.join(get_blob_logdir(), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Accumulator for wandb metrics (logged once per step)
        self.wandb_log_dict = {}
        
        # Load existing best losses if resuming
        self._load_best_losses()
        
        self._load_and_sync_parameters()
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # ✅ FIXED: Load optimizer state after creating optimizer
        if resume_step:
            print(f"✅ Resuming from step: {resume_step}")
            self._load_optimizer_state()
        
        if not th.cuda.is_available():
            logger.warn("Training requires CUDA.")

    def _load_best_losses(self):
        """Load best MSE losses from file if it exists"""
        best_losses_file = os.path.join(self.checkpoint_dir, 'best_losses.txt')
        if os.path.exists(best_losses_file):
            try:
                with open(best_losses_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            modality, loss_str = line.strip().split(':')
                            self.best_losses[modality] = float(loss_str)
                print(f"Loaded best MSE losses: {self.best_losses}")
            except Exception as e:
                print(f"Error loading best losses: {e}")
                self.best_losses = {}
        else:
            self.best_losses = {}

    def _save_best_losses(self):
        """Save best MSE losses to file"""
        best_losses_file = os.path.join(self.checkpoint_dir, 'best_losses.txt')
        try:
            with open(best_losses_file, 'w') as f:
                for modality, loss in self.best_losses.items():
                    f.write(f"{modality}:{loss}\n")
        except Exception as e:
            print(f"Error saving best losses: {e}")

    def run_validation(self):
        """
        Run validation loop on val_data with SSIM, PSNR, and inference time.
        Returns average validation MSE loss.
        """
        if self.val_datal is None:
            print("⚠️ No validation data provided, skipping validation")
            return None
        
        # Import metrics
        try:
            from skimage.metrics import structural_similarity as ssim
            from skimage.metrics import peak_signal_noise_ratio as psnr
            metrics_available = True
        except ImportError:
            print("⚠️ scikit-image not available, skipping SSIM/PSNR metrics")
            metrics_available = False
        
        import time
        self.model.eval()
        val_losses = []
        val_ssims = []
        val_psnrs = []
        inference_times = []
        
        with th.no_grad():
            for batch in self.val_datal:
                # Move batch to device
                if self.mode == 'i2i':
                    batch['t1n'] = batch['t1n'].to(dist_util.dev())
                    batch['t1c'] = batch['t1c'].to(dist_util.dev())
                    batch['t2w'] = batch['t2w'].to(dist_util.dev())
                    batch['t2f'] = batch['t2f'].to(dist_util.dev())
                else:
                    batch = batch.to(dist_util.dev())
                
                # Extract input and target modalities
                modalities = ['t1n', 't1c', 't2w', 't2f']
                input_modalities = [m for m in modalities if m != self.contr]
                
                # Stack input modalities (3 modalities as input)
                x_inputs = [batch[m] for m in input_modalities]
                x_input = th.cat(x_inputs, dim=1)  # [B, 3, H, W, D]
                x_target = batch[self.contr]  # [B, 1, H, W, D]
                
                use_wavelet = self.wavelet is not None and self.wavelet != 'null'
                
                # Time ONLY the inference (full sampling)
                inference_start = time.time()
                
                if use_wavelet:
                    # Apply DWT to inputs
                    input_wavelets = []
                    for x_mod in x_inputs:
                        lfc, *hfcs = self.dwt(x_mod)
                        wavelet_components = th.cat([lfc] + list(hfcs), dim=1)
                        input_wavelets.append(wavelet_components)
                    
                    x_input_wavelet = th.cat(input_wavelets, dim=1)  # [B, 24, H/2, W/2, D/2]
                    
                    # Apply DWT to target for loss calculation
                    target_lfc, *target_hfcs = self.dwt(x_target)
                    x_target_wavelet = th.cat([target_lfc] + list(target_hfcs), dim=1)  # [B, 8, H/2, W/2, D/2]
                    
                    # FULL SAMPLING in wavelet space with conditioning
                    # Shape is for the target only (8 channels), conditioning is passed via cond parameter
                    sample_fn = self.diffusion.p_sample_loop
                    pred_wavelet = sample_fn(
                        self.model,
                        (x_input_wavelet.shape[0], 8, *x_input_wavelet.shape[2:]),  # Target shape: [B, 8, H/2, W/2, D/2]
                        clip_denoised=True,
                        progress=False,
                        cond=x_input_wavelet  # Pass conditioning via cond parameter
                    )
                    
                    # Reconstruct to spatial domain
                    pred_lfc = pred_wavelet[:, :1]
                    pred_hfcs = [pred_wavelet[:, i:i+1] for i in range(1, 8)]
                    pred_spatial = self.idwt(pred_lfc, *pred_hfcs)
                    
                    # MSE loss in wavelet space
                    mse_loss = th.nn.functional.mse_loss(pred_wavelet, x_target_wavelet)
                    
                else:
                    # FULL SAMPLING in image space with conditioning
                    sample_fn = self.diffusion.p_sample_loop
                    pred_spatial = sample_fn(
                        self.model,
                        (x_input.shape[0], 1, *x_input.shape[2:]),  # Target shape: [B, 1, H, W, D]
                        clip_denoised=True,
                        progress=False,
                        cond=x_input  # Pass conditioning via cond parameter
                    )
                    
                    # MSE loss in image space
                    mse_loss = th.nn.functional.mse_loss(pred_spatial, x_target)
                
                inference_end = time.time()
                inference_times.append(inference_end - inference_start)
                
                val_losses.append(mse_loss.item())
                
                # Calculate SSIM and PSNR on spatial domain with brain masking
                if metrics_available:
                    try:
                        # Create brain mask from target
                        target_np = x_target.cpu().numpy()
                        pred_np = pred_spatial.cpu().numpy()
                        
                        brain_mask = (target_np > 0.01).astype(np.float32)
                        predicted_masked = pred_np * brain_mask
                        target_masked = target_np * brain_mask
                        
                        # Calculate metrics on first sample in batch
                        pred_vol = predicted_masked[0, 0]
                        targ_vol = target_masked[0, 0]
                        
                        ssim_score = ssim(targ_vol, pred_vol, data_range=1.0)
                        psnr_score = psnr(targ_vol, pred_vol, data_range=1.0)
                        
                        val_ssims.append(ssim_score)
                        val_psnrs.append(psnr_score)
                        
                    except Exception as e:
                        print(f"  Warning: SSIM/PSNR calculation failed: {e}")
        
        # Back to training mode
        self.model.train()
        
        # Compute average metrics
        avg_val_loss = np.mean(val_losses)
        avg_inference_time = np.mean(inference_times)
        
        # Log ONLY essential metrics to wandb
        self.wandb_log_dict.update({
            'val/mse': avg_val_loss,
            'val/mse': avg_val_loss,
            'val/inference_time': avg_inference_time
        })
        
        if metrics_available and len(val_ssims) > 0:
            avg_ssim = np.mean(val_ssims)
            avg_psnr = np.mean(val_psnrs)
            self.wandb_log_dict.update({
                'val/ssim': avg_ssim,
                'val/psnr': avg_psnr
            })
            print(f"📊 Validation: SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.2f}dB, Inf.Time={avg_inference_time:.3f}s")
        else:
            print(f"📊 Validation: MSE={avg_val_loss:.6f}, Inf.Time={avg_inference_time:.3f}s")
        
        # Save best validation checkpoint
        if avg_val_loss < self.best_val_loss:
            old_best = self.best_val_loss
            self.best_val_loss = avg_val_loss
            improvement = old_best - avg_val_loss if old_best < float('inf') else avg_val_loss
            print(f"🎯 NEW BEST VAL LOSS: {avg_val_loss:.6f} (prev: {old_best:.6f}, -{improvement:.6f})")
            
            # Remove old best val checkpoint if exists
            if self.best_val_checkpoint and os.path.exists(self.best_val_checkpoint):
                try:
                    os.remove(self.best_val_checkpoint)
                    opt_path = self.best_val_checkpoint.replace('.pt', '_opt.pt')
                    if os.path.exists(opt_path):
                        os.remove(opt_path)
                except Exception as e:
                    print(f"Error removing old val checkpoint: {e}")
            
            # Save new best val checkpoint
            filename = f"brats_{self.contr}_best_val_{self.sample_schedule}_{self.diffusion_steps}.pt"
            full_save_path = os.path.join(self.checkpoint_dir, filename)
            
            try:
                with bf.BlobFile(full_save_path, "wb") as f:
                    th.save(self.model.state_dict(), f)
                
                self.best_val_checkpoint = full_save_path
                print(f"✅ Saved best val checkpoint: {full_save_path}")
                
                # Save optimizer state
                opt_save_path = os.path.join(self.checkpoint_dir, f"brats_{self.contr}_best_val_{self.sample_schedule}_{self.diffusion_steps}_opt.pt")
                with bf.BlobFile(opt_save_path, "wb") as f:
                    th.save(self.opt.state_dict(), f)
                
                # Upload to W&B if enabled
                if self.save_to_wandb:
                    self.save_checkpoint_to_wandb(full_save_path)
                    self.save_checkpoint_to_wandb(opt_save_path)
                
            except Exception as e:
                print(f"❌ Error saving best val checkpoint: {e}")
        
        return avg_val_loss

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model ...')
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        # Attempt to find optimizer checkpoint based on the main checkpoint's filename
        # This assumes opt checkpoints are saved with a step number
        step_str = str(self.step).zfill(6) # self.step is the resume_step here
        opt_checkpoint_explicit = bf.join(
            bf.dirname(main_checkpoint), f"opt{step_str}.pt"
        )
        
        # Also try 'opt_best_{modality}.pt' if we are resuming from a 'best' model
        # Note: This logic for loading opt checkpoint could be refined based on the 
        # actual checkpoint naming scheme for non-step checkpoints.
        opt_checkpoint_best = bf.join(
             bf.dirname(main_checkpoint), f"opt_best_{self.contr}.pt"
        )
        
        opt_checkpoint = None
        if bf.exists(opt_checkpoint_explicit):
             opt_checkpoint = opt_checkpoint_explicit
        elif bf.exists(opt_checkpoint_best):
             opt_checkpoint = opt_checkpoint_best
        
        if opt_checkpoint:
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        else:
            print(f'no optimizer checkpoint found for step {self.step} at path {bf.dirname(main_checkpoint)}')

    def run_loop(self):
        import time
        total_data_time = 0.0
        total_step_time = 0.0
        total_log_time = 0.0
        total_save_time = 0.0
        start_time = time.time()
        t = time.time()
        
        # ✅ FIXED: Loop condition now correctly uses self.step (which includes resume_step)
        while not self.lr_anneal_steps or self.step < self.lr_anneal_steps:
            t_total = time.time() - t
            t = time.time()
            
            # --- Reset wandb accumulator for this step ---
            self.wandb_log_dict = {'step': self.step}
            
            # --- Data loading ---
            data_load_start = time.time()
            if self.dataset in ['brats']:
                try:
                    batch = next(self.iterdatal)
                    cond = {}
                except StopIteration:
                    self.iterdatal = iter(self.datal)
                    batch = next(self.iterdatal)
                    cond = {}
            data_load_end = time.time()
            data_load_time = data_load_end - data_load_start
            total_data_time += data_load_time

            # --- Move to device ---
            if self.mode=='i2i':
                batch['t1n'] = batch['t1n'].to(dist_util.dev())
                batch['t1c'] = batch['t1c'].to(dist_util.dev())
                batch['t2w'] = batch['t2w'].to(dist_util.dev())
                batch['t2f'] = batch['t2f'].to(dist_util.dev())
            else:
                batch = batch.to(dist_util.dev())

            # --- Model forward/backward ---
            step_proc_start = time.time()
            mse_loss, sample, sample_idwt = self.run_step(batch, cond)
            step_proc_end = time.time()
            step_time = step_proc_end - step_proc_start
            total_step_time += step_time

            names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]

            # --- Logging ---
            log_start = time.time()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('time/load', total_data_time, global_step=self.step)
                self.summary_writer.add_scalar('time/forward', total_step_time, global_step=self.step)
                self.summary_writer.add_scalar('time/total', t_total, global_step=self.step)
                self.summary_writer.add_scalar('metrics/MSE', mse_loss, global_step=self.step)

            # Log per-step times to WandB
            self.wandb_log_dict.update({
                'train/step_time': step_time,
                'train/data_load_time': data_load_time
            })

            if self.step % 200 == 0:
                image_size = sample_idwt.size()[2]
                midplane = sample_idwt[0, 0, :, :, image_size // 2]
                if self.summary_writer is not None:
                    self.summary_writer.add_image('sample/x_0', midplane.unsqueeze(0), global_step=self.step)
                img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                self.wandb_log_dict['sample/x_0'] = wandb.Image(img, caption='sample/x_0')

                # Only visualize wavelet subbands if using wavelet transform
                if self.diffusion.use_freq and sample.size()[1] == 8:
                    image_size = sample.size()[2]
                    for ch in range(8):
                        midplane = sample[0, ch, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('sample/{}'.format(names[ch]), midplane.unsqueeze(0),
                                                        global_step=self.step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        self.wandb_log_dict[f'sample/{names[ch]}'] = wandb.Image(img, caption=f'sample/{names[ch]}')

                if self.mode == 'i2i':
                    if not self.contr == 't1n':
                        image_size = batch['t1n'].size()[2]
                        midplane = batch['t1n'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t1n', midplane.unsqueeze(0), global_step=self.step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        self.wandb_log_dict['source/t1n'] = wandb.Image(img, caption='source/t1n')
                    if not self.contr == 't1c':
                        image_size = batch['t1c'].size()[2]
                        midplane = batch['t1c'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t1c', midplane.unsqueeze(0), global_step=self.step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        self.wandb_log_dict['source/t1c'] = wandb.Image(img, caption='source/t1c')
                    if not self.contr == 't2w':
                        midplane = batch['t2w'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t2w', midplane.unsqueeze(0), global_step=self.step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        self.wandb_log_dict['source/t2w'] = wandb.Image(img, caption='source/t2w')
                    if not self.contr == 't2f':
                        midplane = batch['t2f'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t2f', midplane.unsqueeze(0), global_step=self.step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        self.wandb_log_dict['source/t2f'] = wandb.Image(img, caption='source/t2f')

            log_end = time.time()
            total_log_time += log_end - log_start

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            # --- Saving ---
            if self.step % self.save_interval == 0 and self.step > 0:
                save_start = time.time()
                self.save_if_best(mse_loss)
                save_end = time.time()
                total_save_time += save_end - save_start
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            
            # --- Validation ---
            if self.val_datal is not None and self.step % self.val_interval == 0 and self.step > 0:
                val_start = time.time()
                val_loss = self.run_validation()
                val_end = time.time()
                print(f"✅ Validation at step {self.step}: val_loss={val_loss:.6f} (took {val_end - val_start:.2f}s)")
            
            # --- Special Checkpoint Saving ---
            if self.special_checkpoint_steps is not None and self.step in self.special_checkpoint_steps and self.step not in self.saved_special_checkpoints:
                save_start = time.time()
                self.save_special_checkpoint(self.step, mse_loss)
                save_end = time.time()
                total_save_time += save_end - save_start
                self.saved_special_checkpoints.add(self.step)
            
            # --- Log all accumulated metrics to wandb (once per step) ---
            wandb.log(self.wandb_log_dict, step=self.step)
            
            # --- Increment step AFTER all logging/saving ---
            self.step += 1

            # Print profiling info every log_interval
            if (self.step - 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"[PROFILE] Step {self.step-1}: Data {total_data_time:.2f}s, Step {total_step_time:.2f}s, Log {total_log_time:.2f}s, Save {total_save_time:.2f}s, Total {elapsed:.2f}s")
                # Debug print for MSE loss tracking
                if (self.step - 1) % 500 == 0:
                    best_loss = self.best_losses.get(self.contr, float('inf'))
                    print(f"[MSE] Step {self.step-1}: Current={mse_loss:.4f}, Best={best_loss:.4f} ({self.contr})")
                # Reset counters for next interval
                total_data_time = 0.0
                total_step_time = 0.0
                total_log_time = 0.0
                total_save_time = 0.0

        # Training finished - log total time
        total_training_time = time.time() - start_time
        wandb.log({'train/total_time': total_training_time}, step=self.step)
        print(f"✅ Training completed in {total_training_time/3600:.2f} hours")
        
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save_if_best(mse_loss)

    def save_if_best(self, current_loss):
        """Only save checkpoint if current MSE loss is better than previous best"""
        modality = self.contr
        
        # Initialize if first time (use inf for MSE loss since lower is better)
        if modality not in self.best_losses:
            self.best_losses[modality] = float('inf')
        
        # Check if this is the best MSE loss so far (LOWER is better for MSE)
        is_best = current_loss < self.best_losses[modality]
        
        if is_best:
            old_best = self.best_losses[modality]
            self.best_losses[modality] = current_loss
            
        if is_best and dist.get_rank() == 0:
            improvement = old_best - current_loss if old_best < float('inf') else current_loss
            print(f"🎯 NEW BEST for {modality}! MSE Loss: {current_loss:.4f} (prev: {old_best:.4f}, -{improvement:.4f})")
            
            # Remove old best checkpoint if it exists
            if modality in self.best_checkpoints:
                old_checkpoint = self.best_checkpoints[modality]
                if os.path.exists(old_checkpoint):
                    try:
                        os.remove(old_checkpoint)
                        print(f"Removed old checkpoint: {old_checkpoint}")
                    except Exception as e:
                        print(f"Error removing old checkpoint: {e}")
            
            # ✅ FIXED: Use self.step directly for naming
            # Use self.step - 1 because save_if_best is called after self.step += 1 in run_loop
            step_to_save = self.step - 1 if self.step > 0 else 0
            filename = f"brats_{self.contr}_{step_to_save:06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
            full_save_path = os.path.join(self.checkpoint_dir, filename)
            
            try:
                with bf.BlobFile(full_save_path, "wb") as f:
                    th.save(self.model.state_dict(), f)
                
                self.best_checkpoints[modality] = full_save_path
                print(f"✅ Saved new best checkpoint: {full_save_path}")
                
                # Save best losses to file
                self._save_best_losses()
                
                # Save optimizer state only for current best
                opt_save_path = os.path.join(self.checkpoint_dir, f"opt_best_{modality}.pt")
                with bf.BlobFile(opt_save_path, "wb") as f:
                    th.save(self.opt.state_dict(), f)
                print(f"💾 Saved optimizer state: {opt_save_path}")
                
                # Upload to W&B if enabled
                if self.save_to_wandb:
                    self.save_checkpoint_to_wandb(full_save_path)
                    self.save_checkpoint_to_wandb(opt_save_path)
                
                # Accumulate checkpoint metrics (will be logged at end of step)
                self.wandb_log_dict.update({
                    f"checkpoints/{modality}/best_loss": current_loss,
                    f"checkpoints/{modality}/improvement": improvement
                })
                
            except Exception as e:
                print(f"❌ Error saving checkpoint: {e}")
        else:
            if not is_best:
                current_best = self.best_losses.get(modality, float('inf'))
                if self.step % 100 == 0:
                    print(f"MSE Loss {current_loss:.4f} not better than best {current_best:.4f} for {modality}")

    def save_special_checkpoint(self, iteration, current_loss):
        """Save special checkpoint at specific iterations for experimental purposes"""
        if dist.get_rank() != 0:
            return
            
        modality = self.contr
        
        # ✅ FIXED: iteration IS self.step now, no need for calculation
        filename = f"brats_{modality}_special_{iteration:06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
        full_save_path = os.path.join(self.checkpoint_dir, filename)
        
        try:
            # Save model checkpoint
            with bf.BlobFile(full_save_path, "wb") as f:
                th.save(self.model.state_dict(), f)
            
            print(f"🎯 SPECIAL CHECKPOINT SAVED at iteration {iteration}!")
            print(f"✅ Saved special checkpoint: {full_save_path}")
            print(f"📊 MSE Loss at iteration {iteration}: {current_loss:.4f}")
            
            # Save optimizer state for special checkpoint
            opt_save_path = os.path.join(self.checkpoint_dir, f"opt_special_{modality}_{iteration:06d}.pt")
            with bf.BlobFile(opt_save_path, "wb") as f:
                th.save(self.opt.state_dict(), f)
            print(f"💾 Saved special optimizer state: {opt_save_path}")
            
            # Upload to W&B if enabled
            if self.save_to_wandb:
                try:
                    wandb.save(full_save_path, base_path=self.checkpoint_dir)
                    wandb.save(opt_save_path, base_path=self.checkpoint_dir)
                    print(f"☁️ Uploaded special checkpoint to W&B: {filename}")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to upload special checkpoint to W&B: {e}")
            
            # Accumulate special checkpoint metrics (will be logged at end of step)
            self.wandb_log_dict.update({
                f"special_checkpoints/{modality}/iteration_{iteration}/loss": current_loss,
                f"special_checkpoints/{modality}/saved_at_step": self.step
            })
            
        except Exception as e:
            print(f"❌ Error saving special checkpoint at iteration {iteration}: {e}")

    def save_checkpoint_to_wandb(self, checkpoint_path, checkpoint_name=None):
        """Helper method to save checkpoint to W&B"""
        if not self.save_to_wandb:
            return
            
        try:
            if checkpoint_name is None:
                checkpoint_name = os.path.basename(checkpoint_path)
            
            wandb.save(checkpoint_path, base_path=os.path.dirname(checkpoint_path))
            print(f"☁️ Uploaded checkpoint to W&B: {checkpoint_name}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to upload checkpoint to W&B: {e}")

    def run_step(self, batch, cond, label=None, info=dict()):
        mse_loss, sample, sample_idwt = self.forward_backward(batch, cond, label)

        if self.use_fp16:
            self.grad_scaler.unscale_(self.opt)

        # compute norms
        with th.no_grad():
            param_max_norm = max([p.abs().max().item() for p in self.model.parameters()])
            grad_max_norm = max([p.grad.abs().max().item() for p in self.model.parameters()])
            info['norm/param_max'] = param_max_norm
            info['norm/grad_max'] = grad_max_norm

        if not th.isfinite(th.tensor(mse_loss)):
            print(f"Non-finite MSE loss: {mse_loss}")
            mse_loss = 0.0

        if not th.isfinite(th.tensor(param_max_norm)):
            logger.log(f"Model parameters contain non-finite value {param_max_norm}, entering breakpoint", level=logger.ERROR)
            breakpoint()

        if self.use_fp16:
            #print("Use fp16 ...")
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            info['scale'] = self.grad_scaler.get_scale()
        else:
            self.opt.step()
        self._anneal_lr()
        self.log_step()
        return mse_loss, sample, sample_idwt

    def forward_backward(self, batch, cond, label=None):
        for p in self.model.parameters():
            p.grad = None

        if self.mode == 'i2i':
            batch_size = batch['t1n'].shape[0]
        else:
            batch_size = batch.shape[0]

        # Sample timesteps
        t, weights = self.schedule_sampler.sample(batch_size, dist_util.dev())

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            x_start=batch,
            t=t,
            model_kwargs=cond,
            labels=label,
            mode=self.mode,
            contr=self.contr
        )
        losses1 = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses1[0]["loss"].detach())

        losses = losses1[0]
        sample = losses1[1]
        sample_idwt = losses1[2]

        # Extract individual loss components for logging
        mse_loss = losses.get("mse_loss", 0.0)
        final_loss = losses.get("loss", th.tensor(0.0))

        # Log all loss components
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('loss/MSE', mse_loss, global_step=self.step)
            self.summary_writer.add_scalar('loss/Final', final_loss.item(), global_step=self.step)
            
            # Log wavelet level MSE losses
            if "mse_wav" in losses:
                mse_wav = losses["mse_wav"]
                self.summary_writer.add_scalar('loss/mse_wav_lll', mse_wav[0].item(), global_step=self.step)
                self.summary_writer.add_scalar('loss/mse_wav_llh', mse_wav[1].item(), global_step=self.step)
                self.summary_writer.add_scalar('loss/mse_wav_lhl', mse_wav[2].item(), global_step=self.step)
                self.summary_writer.add_scalar('loss/mse_wav_lhh', mse_wav[3].item(), global_step=self.step)
                self.summary_writer.add_scalar('loss/mse_wav_hll', mse_wav[4].item(), global_step=self.step)
                self.summary_writer.add_scalar('loss/mse_wav_hlh', mse_wav[5].item(), global_step=self.step)
                self.summary_writer.add_scalar('loss/mse_wav_hhl', mse_wav[6].item(), global_step=self.step)
                self.summary_writer.add_scalar('loss/mse_wav_hhh', mse_wav[7].item(), global_step=self.step)

        # Use MSE loss for backpropagation and model saving
        loss = final_loss
        
        # Accumulate loss metrics (will be logged at end of step)
        self.wandb_log_dict.update({
            'train/mse': mse_loss
        })

        # Create weights for MSE loss
        if "mse_wav" in losses:
            weights = th.ones(len(losses["mse_wav"])).cuda()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items() if k == "mse_wav"})
        else:
            log_loss_dict(self.diffusion, t, {"mse_loss": mse_loss})

        # perform some finiteness checks
        if not th.isfinite(loss):
            logger.log(f"Encountered non-finite MSE loss {loss}")
        if self.use_fp16:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        return mse_loss, sample, sample_idwt

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        # ✅ FIXED: Just use self.step
        frac_done = self.step / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        # ✅ FIXED: Just use self.step
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self):
        """Legacy save method - kept for compatibility but prints warning"""
        print("⚠️ Warning: Using legacy save(). Consider using save_if_best() instead.")
        def save_checkpoint(rate, state_dict):
            if dist.get_rank() == 0:
                logger.log("Saving model...")
                # Compose filename with modality, iterations, sample method, and timesteps
                if self.dataset == 'brats':
                    filename = f"brats_{self.contr}_{self.step:06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
                elif self.dataset == 'lidc-idri':
                    filename = f"lidc-idri_{self.contr}_{self.step:06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
                elif self.dataset == 'brats_inpainting':
                    filename = f"brats_inpainting_{self.contr}_{self.step:06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
                elif self.dataset == 'synthrad':
                    filename = f"synthrad_{self.contr}_{self.step:06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
                else:
                    raise ValueError(f'dataset {self.dataset} not implemented')

                # Create checkpoints directory in /data/
                checkpoint_dir = os.path.join(get_blob_logdir(), 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                full_save_path = os.path.join(checkpoint_dir, filename)
                logger.log(f"Saving model to: {full_save_path}")
                print(f"Saving model to: {full_save_path}")

                with bf.BlobFile(full_save_path, "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.model.state_dict())

        if dist.get_rank() == 0:
            checkpoint_dir = os.path.join(get_blob_logdir(), 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            opt_save_path = os.path.join(checkpoint_dir, f"opt{self.step:06d}.pt")
            print(f"Saving optimizer to: {opt_save_path}")
            
            with bf.BlobFile(opt_save_path, "wb") as f:
                th.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = os.path.basename(filename)
    split = split.split(".")[-2]
    split = split.split("_")[-1]
    # extract trailing number
    reversed_split = []
    for c in reversed(split):
        if not c.isdigit():
            break
        reversed_split.append(c)
    split = ''.join(reversed(reversed_split))
    split = ''.join(c for c in split if c.isdigit())
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    """
    Modified to save checkpoints to /data/ directory where persistent volume is mounted
    """
    return "/data"


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        if hasattr(values, 'mean'):
            logger.logkv_mean(key, values.mean().item())
            # Log the quantiles (four quartiles, in particular).
            if hasattr(values, 'detach'):
                for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                    quartile = int(4 * sub_t / diffusion.num_timesteps)
                    logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
        else:
            # Handle scalar values (like hybrid_loss)
            logger.logkv_mean(key, values.item() if hasattr(values, 'item') else values)


class DirectRegressionLoop(TrainLoop):
    """
    Direct regression training loop (no diffusion).
    Single forward pass: input_modalities → target_modality
    
    Key differences from diffusion:
    - No timestep sampling
    - No noise addition
    - Direct MSE loss in spatial or wavelet domain
    - Much faster training and inference
    """
    
    def __init__(self, *args, **kwargs):
        # Set flag to skip default schedule_sampler creation
        self._skip_schedule_sampler = True
        
        # Remove schedule_sampler from kwargs if present (should be None anyway)
        if 'schedule_sampler' in kwargs:
            kwargs.pop('schedule_sampler')
        
        super().__init__(*args, **kwargs)
        
        # Ensure schedule_sampler is None for direct mode
        self.schedule_sampler = None
        
        print("🚀 Initialized DirectRegressionLoop (no diffusion)")
        print(f"   - Wavelet: {self.wavelet}")
        print(f"   - Target modality: {self.contr}")
        print(f"   - Use wavelet space: {self.wavelet is not None and self.wavelet != 'null'}")
    
    def forward_backward(self, batch, cond, label=None):
        """
        Direct forward pass without diffusion.
        
        Args:
            batch: Dict with modality keys {'t1n', 't1c', 't2w', 't2f'}
            cond: Same as batch (for compatibility)
            label: Unused (for compatibility)
        
        Returns:
            mse_loss: Scalar loss value
            pred_spatial: Predicted image in spatial domain
            pred_spatial: Same (for compatibility with 3-return format)
        """
        for p in self.model.parameters():
            p.grad = None
        
        # Extract input and target modalities
        modalities = ['t1n', 't1c', 't2w', 't2f']
        input_modalities = [m for m in modalities if m != self.contr]
        
        # Stack input modalities (3 modalities as input)
        x_inputs = [batch[m] for m in input_modalities]
        x_input = th.cat(x_inputs, dim=1)  # [B, 3, H, W, D]
        
        # Target modality (1 modality as target)
        x_target = batch[self.contr]  # [B, 1, H, W, D]
        
        use_wavelet = self.wavelet is not None and self.wavelet != 'null'
        
        if use_wavelet:
            # Apply DWT to inputs and target
            # Input: 3 modalities × 8 components = 24 channels
            input_wavelets = []
            for x_mod in x_inputs:
                lfc, *hfcs = self.dwt(x_mod)
                # Concatenate all 8 components
                wavelet_components = th.cat([lfc] + list(hfcs), dim=1)
                input_wavelets.append(wavelet_components)
            
            x_input_wavelet = th.cat(input_wavelets, dim=1)  # [B, 24, H/2, W/2, D/2]
            
            # Target: 1 modality × 8 components = 8 channels
            target_lfc, *target_hfcs = self.dwt(x_target)
            x_target_wavelet = th.cat([target_lfc] + list(target_hfcs), dim=1)  # [B, 8, H/2, W/2, D/2]
            
            # Forward pass in wavelet space
            # Model expects: (x, timesteps=None, **model_kwargs)
            # For direct regression, pass dummy timesteps
            dummy_t = th.zeros(x_input_wavelet.shape[0], dtype=th.long, device=x_input_wavelet.device)
            pred_wavelet = self.model(x_input_wavelet, dummy_t)  # [B, 8, H/2, W/2, D/2]
            
            # MSE loss in wavelet space
            mse_loss = th.nn.functional.mse_loss(pred_wavelet, x_target_wavelet)
            
            # IDWT for visualization and evaluation
            # Split predicted wavelets back into 8 components
            pred_components = th.chunk(pred_wavelet, 8, dim=1)
            pred_lfc = pred_components[0]
            pred_hfcs = pred_components[1:]
            pred_spatial = self.idwt(pred_lfc, *pred_hfcs)
            
        else:
            # Direct prediction in image space (no wavelets)
            # Model expects: (x, timesteps, **model_kwargs)
            dummy_t = th.zeros(x_input.shape[0], dtype=th.long, device=x_input.device)
            pred_spatial = self.model(x_input, dummy_t)  # [B, 1, H, W, D]
            
            # MSE loss in image space
            mse_loss = th.nn.functional.mse_loss(pred_spatial, x_target)
        
        # Backward pass
        loss = mse_loss
        
        if not th.isfinite(loss):
            logger.log(f"Encountered non-finite loss {loss}")
            loss = th.tensor(0.0, device=loss.device, requires_grad=True)
        
        if self.use_fp16:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Accumulate metrics for wandb (will be logged at end of step)
        self.wandb_log_dict.update({
            'train/mse': mse_loss.item()
        })
        
        # Return same format as diffusion loop (3 values)
        return mse_loss.item(), pred_spatial, pred_spatial
    
    def run_validation(self):
        """
        Run validation loop for direct regression with SSIM, PSNR, and inference time.
        Returns average validation MSE loss.
        """
        if self.val_datal is None:
            print("⚠️ No validation data provided, skipping validation")
            return None
        
        # Import metrics
        try:
            from skimage.metrics import structural_similarity as ssim
            from skimage.metrics import peak_signal_noise_ratio as psnr
            metrics_available = True
        except ImportError:
            print("⚠️ scikit-image not available, skipping SSIM/PSNR metrics")
            metrics_available = False
        
        import time
        self.model.eval()
        val_losses = []
        val_ssims = []
        val_psnrs = []
        inference_times = []
        
        with th.no_grad():
            for batch in self.val_datal:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], th.Tensor):
                        batch[key] = batch[key].to(dist_util.dev())
                
                # Extract input and target modalities
                modalities = ['t1n', 't1c', 't2w', 't2f']
                input_modalities = [m for m in modalities if m != self.contr]
                
                # Stack input modalities
                x_inputs = [batch[m] for m in input_modalities]
                x_input = th.cat(x_inputs, dim=1)  # [B, 3, H, W, D]
                x_target = batch[self.contr]  # [B, 1, H, W, D]
                
                use_wavelet = self.wavelet is not None and self.wavelet != 'null'
                
                # Time ONLY the inference (forward pass)
                inference_start = time.time()
                
                if use_wavelet:
                    # Apply DWT to inputs and target
                    input_wavelets = []
                    for x_mod in x_inputs:
                        lfc, *hfcs = self.dwt(x_mod)
                        wavelet_components = th.cat([lfc] + list(hfcs), dim=1)
                        input_wavelets.append(wavelet_components)
                    
                    x_input_wavelet = th.cat(input_wavelets, dim=1)
                    
                    target_lfc, *target_hfcs = self.dwt(x_target)
                    x_target_wavelet = th.cat([target_lfc] + list(target_hfcs), dim=1)
                    
                    # Forward pass
                    dummy_t = th.zeros(x_input_wavelet.shape[0], dtype=th.long, device=x_input_wavelet.device)
                    pred_wavelet = self.model(x_input_wavelet, dummy_t)
                    
                    # Reconstruct to spatial domain for metrics
                    pred_lfc = pred_wavelet[:, :1]
                    pred_hfcs = [pred_wavelet[:, i:i+1] for i in range(1, 8)]
                    pred_spatial = self.idwt(pred_lfc, *pred_hfcs)
                    
                    # MSE loss in wavelet space
                    mse_loss = th.nn.functional.mse_loss(pred_wavelet, x_target_wavelet)
                else:
                    # Forward pass in image space
                    dummy_t = th.zeros(x_input.shape[0], dtype=th.long, device=x_input.device)
                    pred_spatial = self.model(x_input, dummy_t)
                    
                    # MSE loss in image space
                    mse_loss = th.nn.functional.mse_loss(pred_spatial, x_target)
                
                inference_end = time.time()
                inference_times.append(inference_end - inference_start)
                
                val_losses.append(mse_loss.item())
                
                # Calculate SSIM and PSNR on spatial domain with brain masking
                if metrics_available:
                    try:
                        # Create brain mask from target
                        target_np = x_target.cpu().numpy()
                        pred_np = pred_spatial.cpu().numpy()
                        
                        brain_mask = (target_np > 0.01).astype(np.float32)
                        predicted_masked = pred_np * brain_mask
                        target_masked = target_np * brain_mask
                        
                        # Calculate metrics on first sample in batch
                        pred_vol = predicted_masked[0, 0]
                        targ_vol = target_masked[0, 0]
                        
                        ssim_score = ssim(targ_vol, pred_vol, data_range=1.0)
                        psnr_score = psnr(targ_vol, pred_vol, data_range=1.0)
                        
                        val_ssims.append(ssim_score)
                        val_psnrs.append(psnr_score)
                        
                    except Exception as e:
                        print(f"  Warning: SSIM/PSNR calculation failed: {e}")
        
        # Back to training mode
        self.model.train()
        
        # Compute average metrics
        avg_val_loss = np.mean(val_losses)
        avg_inference_time = np.mean(inference_times)
        
        # Log ONLY essential metrics to wandb
        self.wandb_log_dict.update({
            'val/mse': avg_val_loss,
            'val/mse': avg_val_loss,
            'val/inference_time': avg_inference_time
        })
        
        if metrics_available and len(val_ssims) > 0:
            avg_ssim = np.mean(val_ssims)
            avg_psnr = np.mean(val_psnrs)
            self.wandb_log_dict.update({
                'val/ssim': avg_ssim,
                'val/psnr': avg_psnr
            })
            print(f"📊 Validation: SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.2f}dB, Inf.Time={avg_inference_time:.3f}s")
        else:
            print(f"📊 Validation: MSE={avg_val_loss:.6f}, Inf.Time={avg_inference_time:.3f}s")
        
        # Save best validation checkpoint
        if avg_val_loss < self.best_val_loss:
            old_best = self.best_val_loss
            self.best_val_loss = avg_val_loss
            improvement = old_best - avg_val_loss if old_best < float('inf') else avg_val_loss
            print(f"🎯 NEW BEST VAL LOSS: {avg_val_loss:.6f} (prev: {old_best:.6f}, -{improvement:.6f})")
            
            # Remove old best val checkpoint if exists
            if self.best_val_checkpoint and os.path.exists(self.best_val_checkpoint):
                try:
                    os.remove(self.best_val_checkpoint)
                    opt_path = self.best_val_checkpoint.replace('.pt', '_opt.pt')
                    if os.path.exists(opt_path):
                        os.remove(opt_path)
                except Exception as e:
                    print(f"Error removing old val checkpoint: {e}")
            
            # Save new best val checkpoint
            filename = f"brats_{self.contr}_best_val_direct.pt"
            full_save_path = os.path.join(self.checkpoint_dir, filename)
            
            try:
                with bf.BlobFile(full_save_path, "wb") as f:
                    th.save(self.model.state_dict(), f)
                
                self.best_val_checkpoint = full_save_path
                print(f"✅ Saved best val checkpoint: {full_save_path}")
                
                # Save optimizer state
                opt_save_path = os.path.join(self.checkpoint_dir, f"brats_{self.contr}_best_val_direct_opt.pt")
                with bf.BlobFile(opt_save_path, "wb") as f:
                    th.save(self.opt.state_dict(), f)
                
                # Upload to W&B if enabled
                if self.save_to_wandb:
                    self.save_checkpoint_to_wandb(full_save_path)
                    self.save_checkpoint_to_wandb(opt_save_path)
                
            except Exception as e:
                print(f"❌ Error saving best val checkpoint: {e}")
        
        return avg_val_loss
    
    def run_loop(self):
        """
        Main training loop for direct regression.
        Simplified version without diffusion sampling.
        """
        print(f"🏃 Starting DirectRegressionLoop for {self.contr}")
        print(f"   Total steps planned: {self.lr_anneal_steps if self.lr_anneal_steps else 'Based on data iterations'}")
        
        import time
        total_data_time = 0.0
        total_step_time = 0.0
        start_time = time.time()
        t = time.time()
        
        while True:
            t_total = time.time() - t
            t = time.time()
            
            # Initialize wandb accumulator for this step
            self.wandb_log_dict = {'step': self.step}
            
            # --- Data loading ---
            data_load_start = time.time()
            try:
                batch = next(self.iterdatal)
            except StopIteration:
                # Restart the iterator
                self.iterdatal = iter(self.datal)
                batch = next(self.iterdatal)
            data_load_end = time.time()
            data_load_time = data_load_end - data_load_start
            total_data_time += data_load_time
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], th.Tensor):
                    batch[key] = batch[key].to(dist_util.dev())
            
            # --- Model forward/backward ---
            step_proc_start = time.time()
            cond = batch  # For compatibility
            mse_loss, sample, sample_idwt = self.run_step(batch, cond)
            step_proc_end = time.time()
            step_time = step_proc_end - step_proc_start
            total_step_time += step_time
            
            # Log per-step times to WandB
            self.wandb_log_dict.update({
                'train/step_time': step_time,
                'train/data_load_time': data_load_time
            })
            
            # Logging
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
                print(f"Step {self.step}: MSE Loss = {mse_loss:.6f}")
                
                # Print profiling info
                if self.step > 0:
                    elapsed = time.time() - start_time
                    print(f"[PROFILE] Step {self.step}: DataLoad {total_data_time:.2f}s, StepTime {total_step_time:.2f}s, Elapsed {elapsed:.2f}s")
            
            # Save checkpoints
            if self.step % self.save_interval == 0 and self.step > 0:
                self.save_if_best(mse_loss)
            
            # Validation
            if self.val_datal is not None and self.step % self.val_interval == 0 and self.step > 0:
                val_loss = self.run_validation()
                print(f"✅ Validation at step {self.step}: val_loss={val_loss:.6f}")
            
            # Check for special checkpoints
            if self.special_checkpoint_steps and self.step in self.special_checkpoint_steps:
                if self.step not in self.saved_special_checkpoints:
                    self.save_special_checkpoint(self.step, mse_loss)
                    self.saved_special_checkpoints.add(self.step)
            
            # Log all accumulated metrics to wandb (once per step)
            wandb.log(self.wandb_log_dict, step=self.step)
            
            # Increment step AFTER all logging/saving
            self.step += 1
            
            # Optional: Stop after certain number of iterations
            if self.lr_anneal_steps and self.step > self.lr_anneal_steps:
                print(f"✅ Reached {self.lr_anneal_steps} steps, stopping training")
                break
        
        # Training finished - log total time
        total_training_time = time.time() - start_time
        wandb.log({'train/total_time': total_training_time}, step=self.step)
        print(f"✅ Training completed in {total_training_time/3600:.2f} hours")