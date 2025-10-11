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
            save_to_wandb=True
        ):
            self.summary_writer = summary_writer
            self.mode = mode
            self.model = model
            self.diffusion = diffusion
            self.datal = data
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
            self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
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
            self.dwt = DWT_3D(self.wavelet)
            self.idwt = IDWT_3D(self.wavelet)
            
            # Training state
            self.step = 1
            self.resume_step = resume_step
            self.global_batch = self.batch_size * dist.get_world_size()
            self.sync_cuda = th.cuda.is_available()
            
            # Track best MSE loss (lower is better)
            self.best_losses = {}
            self.best_checkpoints = {}
            self.checkpoint_dir = os.path.join(get_blob_logdir(), 'checkpoints')
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            # Load existing best losses if resuming
            self._load_best_losses()
            
            self._load_and_sync_parameters()
            self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            if self.resume_step:
                print("Resume Step: " + str(self.resume_step))
                self._load_optimizer_state()
            if not th.cuda.is_available():
                logger.warn(
                    "Training requires CUDA. "
                )

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
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        else:
            print('no optimizer checkpoint exists')

    def run_loop(self):
        import time
        total_data_time = 0.0
        total_step_time = 0.0
        total_log_time = 0.0
        total_save_time = 0.0
        start_time = time.time()
        t = time.time()
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            t_total = time.time() - t
            t = time.time()
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
            total_data_time += data_load_end - data_load_start

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
            total_step_time += step_proc_end - step_proc_start

            names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]

            # --- Logging ---
            log_start = time.time()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('time/load', total_data_time, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/forward', total_step_time, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/total', t_total, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('metrics/MSE', mse_loss, global_step=self.step + self.resume_step)

            wandb_log_dict = {
                'time/load': total_data_time,
                'time/forward': total_step_time,
                'time/total': t_total,
                'metrics/MSE': mse_loss,
                'step': self.step + self.resume_step
            }

            if self.step % 200 == 0:
                image_size = sample_idwt.size()[2]
                midplane = sample_idwt[0, 0, :, :, image_size // 2]
                if self.summary_writer is not None:
                    self.summary_writer.add_image('sample/x_0', midplane.unsqueeze(0),
                                                  global_step=self.step + self.resume_step)
                img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                wandb_log_dict['sample/x_0'] = wandb.Image(img, caption='sample/x_0')

                image_size = sample.size()[2]
                for ch in range(8):
                    midplane = sample[0, ch, :, :, image_size // 2]
                    if self.summary_writer is not None:
                        self.summary_writer.add_image('sample/{}'.format(names[ch]), midplane.unsqueeze(0),
                                                      global_step=self.step + self.resume_step)
                    img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                    wandb_log_dict[f'sample/{names[ch]}'] = wandb.Image(img, caption=f'sample/{names[ch]}')

                if self.mode == 'i2i':
                    if not self.contr == 't1n':
                        image_size = batch['t1n'].size()[2]
                        midplane = batch['t1n'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t1n', midplane.unsqueeze(0),
                                                          global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t1n'] = wandb.Image(img, caption='source/t1n')
                    if not self.contr == 't1c':
                        image_size = batch['t1c'].size()[2]
                        midplane = batch['t1c'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t1c', midplane.unsqueeze(0),
                                                          global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t1c'] = wandb.Image(img, caption='source/t1c')
                    if not self.contr == 't2w':
                        midplane = batch['t2w'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t2w', midplane.unsqueeze(0),
                                                          global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t2w'] = wandb.Image(img, caption='source/t2w')
                    if not self.contr == 't2f':
                        midplane = batch['t2f'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t2f', midplane.unsqueeze(0),
                                                          global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t2f'] = wandb.Image(img, caption='source/t2f')

            wandb.log(wandb_log_dict, step=self.step + self.resume_step)
            log_end = time.time()
            total_log_time += log_end - log_start

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            # --- Saving ---
            if self.step % self.save_interval == 0:
                save_start = time.time()
                self.save_if_best(mse_loss)
                save_end = time.time()
                total_save_time += save_end - save_start
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            
            # --- Special Checkpoint Saving ---
            current_iteration = self.step + self.resume_step
            if current_iteration in self.special_checkpoint_steps and current_iteration not in self.saved_special_checkpoints:
                save_start = time.time()
                self.save_special_checkpoint(current_iteration, mse_loss)
                save_end = time.time()
                total_save_time += save_end - save_start
                self.saved_special_checkpoints.add(current_iteration)
            
            self.step += 1

            # Print profiling info every log_interval
            if self.step % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"[PROFILE] Step {self.step}: Data {total_data_time:.2f}s, Step {total_step_time:.2f}s, Log {total_log_time:.2f}s, Save {total_save_time:.2f}s, Total {elapsed:.2f}s")
                # Debug print for MSE loss tracking
                if self.step <= 1000 or self.step % 500 == 0:
                    best_loss = self.best_losses.get(self.contr, float('inf'))
                    print(f"[MSE] Step {self.step}: Current={mse_loss:.4f}, Best={best_loss:.4f} ({self.contr})")
                # Reset counters for next interval
                total_data_time = 0.0
                total_step_time = 0.0
                total_log_time = 0.0
                total_save_time = 0.0

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
            
            # Save new best checkpoint
            filename = f"brats_{self.contr}_{(self.step+self.resume_step):06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
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
                
                # Log to wandb
                wandb.log({
                    f"checkpoints/{modality}/best_loss": current_loss,
                    f"checkpoints/{modality}/improvement": improvement,
                    "step": self.step + self.resume_step
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
        
        # Create special checkpoint filename
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
            
            # Log special checkpoint metrics to wandb
            wandb.log({
                f"special_checkpoints/{modality}/iteration_{iteration}/loss": current_loss,
                f"special_checkpoints/{modality}/saved_at_step": self.step + self.resume_step,
                "step": self.step + self.resume_step
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
            print("Use fp16 ...")
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
            self.summary_writer.add_scalar('loss/MSE', mse_loss, global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/Final', final_loss.item(), global_step=self.step + self.resume_step)
            
            # Log wavelet level MSE losses
            if "mse_wav" in losses:
                mse_wav = losses["mse_wav"]
                self.summary_writer.add_scalar('loss/mse_wav_lll', mse_wav[0].item(), global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/mse_wav_llh', mse_wav[1].item(), global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/mse_wav_lhl', mse_wav[2].item(), global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/mse_wav_lhh', mse_wav[3].item(), global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/mse_wav_hll', mse_wav[4].item(), global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/mse_wav_hlh', mse_wav[5].item(), global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/mse_wav_hhl', mse_wav[6].item(), global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/mse_wav_hhh', mse_wav[7].item(), global_step=self.step + self.resume_step)

        # Use MSE loss for backpropagation and model saving
        loss = final_loss
        
        # Add to wandb logging
        wandb.log({
            'loss/MSE': mse_loss,
            'loss/Final': final_loss.item(),
            'step': self.step + self.resume_step
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
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        """Legacy save method - kept for compatibility but prints warning"""
        print("⚠️  Warning: Using legacy save(). Consider using save_if_best() instead.")
        def save_checkpoint(rate, state_dict):
            if dist.get_rank() == 0:
                logger.log("Saving model...")
                # Compose filename with modality, iterations, sample method, and timesteps
                if self.dataset == 'brats':
                    filename = f"brats_{self.contr}_{(self.step+self.resume_step):06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
                elif self.dataset == 'lidc-idri':
                    filename = f"lidc-idri_{self.contr}_{(self.step+self.resume_step):06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
                elif self.dataset == 'brats_inpainting':
                    filename = f"brats_inpainting_{self.contr}_{(self.step + self.resume_step):06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
                elif self.dataset == 'synthrad':
                    filename = f"synthrad_{self.contr}_{(self.step + self.resume_step):06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
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
            opt_save_path = os.path.join(checkpoint_dir, f"opt{(self.step+self.resume_step):06d}.pt")
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