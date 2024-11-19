
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import wandb
import os
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding_trainer")
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.sum((anchor - positive) ** 2, dim=1)
        distance_negative = torch.sum((anchor - negative) ** 2, dim=1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        
        # Debug first sample in batch
        if torch.distributed.get_rank() == 0:
            with torch.no_grad():
                pos_dist = distance_positive[0].item()
                neg_dist = distance_negative[0].item()
                loss_val = losses[0].item()
                print(f"\n[Sample Distances] Pos: {pos_dist:.4f}, Neg: {neg_dist:.4f}, Loss: {loss_val:.4f}")
        
        return losses.mean()

class EmbeddingTrainer:
    def train(self, train_dataloader, val_dataloader, 
              num_epochs=3, 
              learning_rate=2e-5,
              warmup_steps=0,
              logging_steps=100,
              rank=0):
              
        if self.optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
                
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        for epoch in range(num_epochs):
            train_dataloader.sampler.set_epoch(epoch)
            
            if rank == 0:
                print(f"\n{'='*20} Epoch {epoch + 1}/{num_epochs} {'='*20}")
            
            self.model.train()
            epoch_train_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", disable=(rank != 0))

            for step, batch in enumerate(progress_bar):
                try:
                    # Get embeddings
                    anchor_emb = self.compute_embeddings(batch, 'anchor')
                    positive_emb = self.compute_embeddings(batch, 'positive')
                    negative_emb = self.compute_embeddings(batch, 'negative')

                    # Print embedding norms occasionally
                    if rank == 0 and step % logging_steps == 0:
                        print(f"\n[Step {step} Norms]")
                        print(f"Anchor: {torch.norm(anchor_emb[0]).item():.4f}")
                        print(f"Positive: {torch.norm(positive_emb[0]).item():.4f}")
                        print(f"Negative: {torch.norm(negative_emb[0]).item():.4f}")

                    # Compute loss
                    loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    epoch_train_loss += loss.item()

                    # Print similarities and update progress
                    if rank == 0 and step % logging_steps == 0:
                        with torch.no_grad():
                            pos_sim = F.cosine_similarity(
                                anchor_emb[0:1], positive_emb[0:1]
                            ).item()
                            neg_sim = F.cosine_similarity(
                                anchor_emb[0:1], negative_emb[0:1]
                            ).item()
                            print(f"[Similarities] Pos: {pos_sim:.4f}, Neg: {neg_sim:.4f}")
                            print(f"Current loss: {loss.item():.4f}")
                            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.2e}")

                        wandb.log({
                            'epoch': epoch + 1,
                            'step': step,
                            'train_loss': loss.item(),
                            'learning_rate': self.scheduler.get_last_lr()[0],
                            'positive_similarity': pos_sim,
                            'negative_similarity': neg_sim
                        })

                    progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

                except Exception as e:
                    print(f"Error in training step: {str(e)}")
                    continue

            avg_train_loss = epoch_train_loss / len(train_dataloader)
            
            if rank == 0:
                print(f"\n{'='*20} Epoch {epoch + 1} Summary {'='*20}")
                print(f"Average training loss: {avg_train_loss:.4f}")
                
                val_loss = self.evaluate(val_dataloader)
                print(f"Validation loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model("best_model")
                    print("\n★ New best model saved! ★")

                wandb.log({
                    'epoch': epoch + 1,
                    'avg_train_loss': avg_train_loss,
                    'val_loss': val_loss,
                })

    @torch.no_grad()
    def evaluate(self, dataloader, rank=0):
        self.model.eval()
        total_loss = 0
        all_pos_sims = []
        all_neg_sims = []
        
        for batch in tqdm(dataloader, desc="Evaluating", disable=(rank != 0)):
            try:
                anchor_emb = self.compute_embeddings(batch, 'anchor')
                positive_emb = self.compute_embeddings(batch, 'positive')
                negative_emb = self.compute_embeddings(batch, 'negative')

                # Calculate similarities for statistics
                pos_sims = F.cosine_similarity(anchor_emb, positive_emb)
                neg_sims = F.cosine_similarity(anchor_emb, negative_emb)
                all_pos_sims.extend(pos_sims.cpu().tolist())
                all_neg_sims.extend(neg_sims.cpu().tolist())

                loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)
                total_loss += loss.item()

            except Exception as e:
                print(f"Error in evaluation step: {str(e)}")
                continue

        if rank == 0:
            print("\n=== Validation Statistics ===")
            print(f"Average positive similarity: {np.mean(all_pos_sims):.4f}")
            print(f"Average negative similarity: {np.mean(all_neg_sims):.4f}")
            print(f"Positive similarity range: [{min(all_pos_sims):.4f}, {max(all_pos_sims):.4f}]")
            print(f"Negative similarity range: [{min(all_neg_sims):.4f}, {max(all_neg_sims):.4f}]")

        return total_loss / len(dataloader)
def setup(rank, world_size):
    # Removed environment variable settings from here
    # They will be set in the main process
    # Initialize process group
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()
def main_worker(rank, world_size, args):
    try:
        setup(rank, world_size)

        if rank == 0:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
            wandb.init(project=args.wandb_project)
            wandb.config.update(args)
            print("\n=== Starting Training ===")
            print(f"Number of GPUs: {world_size}")
            print(f"Model: {args.model_name}")
            print(f"Batch size per GPU: {args.batch_size}")
            print(f"Learning rate: {args.learning_rate}")

        # Rest of the main_worker code remains the same...
        # Set seeds for reproducibility
        print("Seeding")
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)  
        print("seeded")
        # Load data
        df = pd.read_csv(args.data_path)
        questions = df['question'].tolist()
        positive_contexts = df['context'].tolist()
        all_contexts = df['context'].unique().tolist()
        print("Data Loaded")
        # Create negative samples
        # For each question, pair it with one positive context and multiple negative contexts
        # This will transform the data into a classification format

        # Parameters
        num_negatives = args.num_negatives

        questions_list = []
        contexts_list = []
        labels_list = []

        for q, pos_c in zip(questions, positive_contexts):
            # Positive sample
            questions_list.append(q)
            contexts_list.append(pos_c)
            labels_list.append(1)  # Positive label

            # Negative samples
            neg_contexts = np.random.choice([c for c in all_contexts if c != pos_c], size=num_negatives, replace=False)
            for neg_c in neg_contexts:
                questions_list.append(q)
                contexts_list.append(neg_c)
                labels_list.append(0)  # Negative label
        print("Samples generated")

        # Split data
        train_questions, val_questions, train_contexts, val_contexts, train_labels, val_labels = train_test_split(
            questions_list,
            contexts_list,
            labels_list,
            test_size=args.train_test_split,
            random_state=args.random_seed,
            stratify=labels_list
        )
        print("Data splitted")

        # Initialize tokenizer and datasets
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        train_dataset = ClassificationDataset(
            questions=train_questions,
            contexts=train_contexts,
            labels=train_labels,
            tokenizer=tokenizer,
            max_length=args.max_length
        )

        val_dataset = ClassificationDataset(
            questions=val_questions,
            contexts=val_contexts,
            labels=val_labels,
            tokenizer=tokenizer,
            max_length=args.max_length
        )
        print("Dataset created")

        # Create samplers
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        print("Sampler created")

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True  # Important for DDP
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False  # Typically, no need to drop last batch in validation
        )
        print("Loader created")

        trainer = EmbeddingTrainer(
            model_name=args.model_name,
            device=torch.device(f'cuda:{rank}'),
            pooling_strategy=args.pooling_strategy,
            embedding_dim=args.embedding_dim
        )
        print("Trainer created")

        # Wrap model in DDP
        trainer.model = DDP(
            trainer.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False  # Better performance
        )
        print("DDP wrapped")

        # Train model
        print("trainning starts")

        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            rank=rank
        )
        print("finished train")

    except Exception as e:
        logger.error(f"Error in main_worker: {str(e)}")
        raise e
    finally:
        cleanup()

def parse_args():
    parser = argparse.ArgumentParser(description="DDP Training for Sentence Embeddings with Cross Entropy Loss")
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                      help='Pretrained model name')
    parser.add_argument('--pooling_strategy', type=str, default='cls',
                      choices=['cls', 'mean'],
                      help='Pooling strategy for embeddings')
    parser.add_argument('--embedding_dim', type=int, default=768,
                      help='Dimension of embeddings')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Per-GPU batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=0,
                      help='Number of warmup steps')
    parser.add_argument('--max_length', type=int, default=512,
                      help='Maximum sequence length')
    parser.add_argument('--num_negatives', type=int, default=1,
                      help='Number of negative samples per positive')
    parser.add_argument('--train_test_split', type=float, default=0.2,
                      help='Train-validation split ratio')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of DataLoader workers')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the training data CSV')
    
    # Logging parameters
    parser.add_argument('--wandb_project', type=str, default='sentence-embeddings',
                      help='WandB project name')
    parser.add_argument('--wandb_api_key', type=str, required=True,
                      help='WandB API Key')
    parser.add_argument('--logging_steps', type=int, default=100,
                      help='Number of steps between logging')
    
    # DDP parameters
    parser.add_argument('--master_addr', type=str, default='127.0.0.1',
                      help='Master node address')
    parser.add_argument('--master_port', type=str, default='29500',
                      help='Master node port')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA to be available")
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No GPUs available for training.")
    
    # Set environment variables before spawning
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    try:
        # Launch processes
        print("HELLO")
        torch.multiprocessing.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise e
