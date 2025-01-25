import os
import torch
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
import wandb
from pytorch_lightning.loggers import WandbLogger
from config import SmolLM2Config
from model import SmolLM2ForCausalLM
import time
import warnings
import logging

# Filter out specific warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers`.*")
warnings.filterwarnings("ignore", ".*The progress bar.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*torch.utils.checkpoint: the use_reentrant.*")
warnings.filterwarnings("ignore", ".*You are using `torch.load` with `weights_only=False`.*")
warnings.filterwarnings("ignore", ".*Checkpoint directory.*exists and is not empty.*")

# Configure logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("wandb").setLevel(logging.ERROR)

class StreamingShakespeareDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, chunk_size=128):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        
    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        tokens = self.tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        
        for i in range(0, len(tokens) - self.chunk_size, self.chunk_size // 2):
            chunk = tokens[i:i + self.chunk_size]
            if len(chunk) == self.chunk_size:
                yield {
                    "input_ids": chunk,
                    "labels": chunk.clone()
                }

class SmolLM2LightningModule(pl.LightningModule):
    def __init__(self, config, start_time=None, total_tokens=0):
        super().__init__()
        self.save_hyperparameters(ignore=['start_time'])
        self.config = config
        self.model = SmolLM2ForCausalLM(config)
        self.current_step = 0
        self.start_time = start_time if start_time is not None else time.time()
        self.last_time = self.start_time
        self.total_tokens = total_tokens
        self.training_step_outputs = []
        self.log_file = open("generation_logs.txt", "a", encoding='utf-8')
        
        # Enable gradient checkpointing
        self.model.model.gradient_checkpointing = True
        
    def on_train_epoch_start(self):
        self.training_step_outputs = []  # Reset at epoch start
        
    def forward(self, input_ids, labels=None):
        return self.model(input_ids=input_ids, labels=labels)
    
    def training_step(self, batch, batch_idx):
        try:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            _, loss = self(input_ids, labels)
            
            # Calculate tokens before storing loss
            tokens_in_batch = input_ids.numel()
            self.total_tokens += tokens_in_batch
            tokens_per_sec = tokens_in_batch / dt if dt > 0 else 0
            elapsed_time = current_time - self.start_time
            
            self.training_step_outputs.append({
                'loss': loss,
                'total_tokens': self.total_tokens
            })
            
            # Log every 50 steps
            if self.current_step % 50 == 0:
                log_str = (
                    f"Step {self.current_step:5d} | "
                    f"Loss: {loss.item():8.4f} | "
                    f"Time: {elapsed_time:8.1f}s | "
                    f"Tokens/sec: {tokens_per_sec:8.2f} | "
                    f"Total Tokens: {self.total_tokens:10d}"
                )
                print(log_str)
                self.log_file.write(log_str + "\n")
                self.log_file.flush()
            
            self.current_step += 1
            return loss
            
        except Exception as e:
            print(f"\nError in training step: {e}")
            self.log_file.write(f"\nError in training step: {e}\n")
            self.log_file.flush()
            raise e

    def configure_optimizers(self):
        # Use a smaller learning rate
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,  # Reduced from 3e-4
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Simpler scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-4,
            total_steps=5000,
            pct_start=0.1,
            anneal_strategy='linear'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def on_train_end(self):
        self.log_file.close()  # Close file when training ends

def generate_sample_text(model, tokenizer, prompt, max_length=30):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        output_sequence = input_ids.clone()
        
        for _ in range(max_length - len(input_ids[0])):
            outputs, _ = model(input_ids=output_sequence)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            output_sequence = torch.cat([output_sequence, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
        return tokenizer.decode(output_sequence[0], skip_special_tokens=True)

class GenerativeTextCallback(pl.Callback):
    def __init__(self, tokenizer, prompts, every_n_steps=500):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.every_n_steps = every_n_steps
        self.last_step = -1
        # Open file with UTF-8 encoding
        self.log_file = open("generation_logs.txt", "a", encoding='utf-8')
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_step = pl_module.current_step
        
        if current_step % self.every_n_steps == 0 and current_step != self.last_step:
            self.last_step = current_step
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Save checkpoint with total_tokens
            checkpoint_path = f"checkpoints/step_{current_step}.pt"
            torch.save({
                'step': current_step,
                'model_state_dict': pl_module.state_dict(),
                'optimizer_state_dict': trainer.optimizers[0].state_dict(),
                'loss': outputs.item() if torch.is_tensor(outputs) else outputs['loss'].item(),
                'total_tokens': pl_module.total_tokens,  # Save total tokens
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
            
            log_str = f"\n=== Generation samples at step {current_step} (Time: {timestamp}) ===\n"
            print(log_str)
            self.log_file.write(log_str)
            
            for prompt in self.prompts:
                generated = generate_sample_text(pl_module, self.tokenizer, prompt)
                sample_str = f"\nPrompt: {prompt}\nGenerated: {generated}\n"
                print(sample_str)
                self.log_file.write(sample_str)
                
                # Log to wandb
                trainer.logger.experiment.log({
                    f"generation_{prompt}": generated,
                    "step": current_step
                })
            
            # Get the current loss from the training step
            current_loss = pl_module.training_step_outputs[-1]['loss'].item() if hasattr(pl_module, 'training_step_outputs') else outputs.item()
            
            # Save training stats
            stats_str = (
                f"\nTraining Stats at step {current_step}:\n"
                f"Loss: {current_loss:.4f}\n"
                f"Learning Rate: {trainer.optimizers[0].param_groups[0]['lr']:.6f}\n"
                f"Total Tokens: {pl_module.total_tokens:,}\n"
                f"Elapsed Time: {time.time() - pl_module.start_time:.1f}s\n"
            )
            print(stats_str)
            self.log_file.write(stats_str)
            
            self.log_file.flush()

    def on_train_end(self, trainer, pl_module):
        self.log_file.close()

def get_latest_checkpoint():
    """Get the latest checkpoint from the checkpoints directory"""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("step_") and f.endswith(".pt")]
    if not checkpoints:
        return None
        
    # Extract step numbers and find the latest
    steps = [int(f.split("_")[1].split(".")[0]) for f in checkpoints]
    latest_step = max(steps)
    return f"checkpoints/step_{latest_step}.pt"

def main():
    pl.seed_everything(42)
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Append to log file instead of overwriting
    with open("generation_logs.txt", "a", encoding='utf-8') as f:
        f.write(f"\n\n{'='*50}\nNew training session started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}\n\n")
    
    config = SmolLM2Config()
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/cosmo2-tokenizer",
        revision=None,
        use_fast=True
    )
    
    test_prompts = [
        "To be or not",
        "All the world's",
        "Friends, Romans,",
        "Now is the winter"
    ]
    
    # Phase 1: Initial 5000 steps or resume from checkpoint
    latest_checkpoint = get_latest_checkpoint()
    starting_step = 0
    
    if latest_checkpoint:
        print(f"\nFound checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        starting_step = checkpoint['step']
        starting_tokens = checkpoint.get('total_tokens', 0)
        print(f"Resuming from step {starting_step} with {starting_tokens:,} tokens")
        
        # Initialize model with existing tokens count
        model = SmolLM2LightningModule(
            config, 
            start_time=time.time(),  # Reset time for accurate timing
            total_tokens=starting_tokens  # Pass existing token count
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.current_step = starting_step
    else:
        print("\nNo checkpoint found, starting from step 0")
        model = SmolLM2LightningModule(config)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Summary:")
    print("-" * 53)
    print(f"| {'Name':^6} | {'Type':^15} | {'Params':^8} | {'Mode':^15} |")
    print("-" * 53)
    print(f"| {'0':^6} | {'SmolLM2ForCausalLM':^15} | {total_params/1e6:>3.0f} M | {'train':^15} |")
    print("-" * 53)
    print(f"{trainable_params/1e6:.0f} M     Trainable params")
    print(f"{(total_params-trainable_params)/1e6:.0f}         Non-trainable params")
    print(f"{total_params/1e6:.0f} M     Total params")
    print(f"{total_params*4/1024/1024:.3f}   Total estimated model params size (MB)")
    print(f"{sum(1 for m in model.modules() if m.training)} Modules in train mode")
    print(f"{sum(1 for m in model.modules() if not m.training)} Modules in eval mode")
    print()

    # Configure wandb to be less verbose
    os.environ["WANDB_SILENT"] = "true"
    
    # Initialize wandb with quiet mode
    wandb_logger = WandbLogger(
        project="smollm2",
        name=f"smollm2-135M-phase1-from-{starting_step}",
        save_dir="logs",
        mode="offline",
        log_model=False  # Reduce wandb logging
    )
    
    # Prepare dataset
    train_dataset = StreamingShakespeareDataset(
        "input.txt", 
        tokenizer,
        chunk_size=64
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False
    )
    
    # Phase 1 callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="step_{step}",
        every_n_train_steps=500,
        save_top_k=-1
    )
    
    generative_callback = GenerativeTextCallback(
        tokenizer=tokenizer,
        prompts=test_prompts,
        every_n_steps=500
    )
    
    # Calculate remaining steps for phase 1
    remaining_steps = 5000 - starting_step
    
    if remaining_steps > 0:
        print(f"\nPhase 1: Training for {remaining_steps} steps (continuing from step {starting_step})...")
        
        # Trainer for phase 1
        trainer = pl.Trainer(
            max_steps=remaining_steps,  # This ensures we stop at exactly 5000 steps
            callbacks=[checkpoint_callback, generative_callback],
            logger=wandb_logger,
            gradient_clip_val=1.0,
            accumulate_grad_batches=64,
            precision='32-true',
            accelerator='cpu',
            devices=1,
            enable_progress_bar=False,
            log_every_n_steps=50,
            enable_model_summary=True,
            deterministic=True,
            strategy='auto'
        )
        
        try:
            trainer.fit(model, train_loader)
            print("\nPhase 1 completed successfully!")
            
            # Force save checkpoint at 5000 steps
            torch.save({
                'step': 5000,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizers[0].state_dict(),
                'total_tokens': model.total_tokens,
            }, "checkpoints/step_5000.pt")
            print("\nSaved final Phase 1 checkpoint")
            
        except KeyboardInterrupt:
            print("\nPhase 1 interrupted! Saving checkpoint...")
            torch.save({
                'step': model.current_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizers[0].state_dict(),
                'total_tokens': model.total_tokens,
            }, f"checkpoints/interrupt_step_{model.current_step}.pt")
            return
    
    # Only proceed to Phase 2 if Phase 1 is complete
    if starting_step >= 5000:
        print("\nPhase 1 already completed. Skipping to Phase 2.")
    elif model.current_step >= 5000:
        print("\nPhase 1 just completed. Moving to Phase 2.")
    else:
        print("\nPhase 1 not complete. Please complete Phase 1 first.")
        return
        
    print("\nStarting Phase 2: Additional 50 steps")
    
    # Load the 5000-step checkpoint
    final_checkpoint = "checkpoints/step_5000.pt"
    if os.path.exists(final_checkpoint):
        checkpoint = torch.load(final_checkpoint)
        model = SmolLM2LightningModule(
            config,
            start_time=time.time(),
            total_tokens=checkpoint.get('total_tokens', 0)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.current_step = 5000
        print("Loaded checkpoint from step 5000")
    else:
        print("Warning: Could not find step 5000 checkpoint!")
        return
    
    # New wandb logger for phase 2
    wandb_logger = WandbLogger(
        project="smollm2",
        name="smollm2-135M-phase2",
        save_dir="logs",
        mode="offline"
    )
    
    # Phase 2 callbacks with exact step limit
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="phase2_step_{step}",
        every_n_train_steps=50,
        save_top_k=-1
    )
    
    class StopAtStepCallback(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if pl_module.current_step >= 5050:  # Stop at exactly 5050
                trainer.should_stop = True
    
    generative_callback = GenerativeTextCallback(
        tokenizer=tokenizer,
        prompts=test_prompts,
        every_n_steps=50
    )
    
    # Trainer for phase 2 with exact step limit
    trainer = pl.Trainer(
        max_steps=50,  # Only 50 more steps
        callbacks=[checkpoint_callback, generative_callback, StopAtStepCallback()],
        logger=wandb_logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=64,
        precision='32-true',
        accelerator='cpu',
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        deterministic=True,
        strategy='auto'
    )
    
    try:
        print("\nStarting additional 50 steps...")
        trainer.fit(model, train_loader)
        print("\nPhase 2 completed successfully!")
        
        # Final generation
        print("\n=== Final generation samples ===")
        for prompt in test_prompts:
            generated = generate_sample_text(model, tokenizer, prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated}")
            
    except Exception as e:
        print(f"\nError in Phase 2: {e}")
        torch.save({
            'step': model.current_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizers[0].state_dict(),
        }, "checkpoints/phase2_error.pt")
        raise e

if __name__ == "__main__":
    main() 