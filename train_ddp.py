import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import wandb  # for experiment tracking
import os
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding_trainer")

class TripletDataset(Dataset):
    def __init__(self, questions, positive_contexts, all_contexts, tokenizer, max_length=512, num_negatives=1, random_seed=42):
        self.questions = questions
        self.positive_contexts = positive_contexts
        self.all_contexts = all_contexts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.random_seed = random_seed
        self.context_to_question = self._build_context_to_question_map()

    def _build_context_to_question_map(self):
        # Create a mapping from context to questions for efficient negative sampling
        context_to_question = {}
        for q, c in zip(self.questions, self.positive_contexts):
            if c not in context_to_question:
                context_to_question[c] = []
            context_to_question[c].append(q)
        return context_to_question

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        anchor_question = self.questions[idx]
        positive_context = self.positive_contexts[idx]

        # Select negative contexts
        negative_contexts = self._get_negative_contexts(idx)

        # Tokenize anchor question
        anchor_encoding = self.tokenizer(
            anchor_question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize positive context
        positive_encoding = self.tokenizer(
            positive_context,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize negative contexts
        negative_encodings = self.tokenizer(
            negative_contexts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(),
            'negative_input_ids': negative_encodings['input_ids'],       # Shape: (num_negatives, max_length)
            'negative_attention_mask': negative_encodings['attention_mask']  # Shape: (num_negatives, max_length)
        }

    def _get_negative_contexts(self, idx):
        # Ensure reproducibility
        np.random.seed(self.random_seed + idx)
        
        # Select random negative contexts that are not the positive context
        negative_contexts = []
        while len(negative_contexts) < self.num_negatives:
            neg_ctx = np.random.choice(self.all_contexts)
            if neg_ctx != self.positive_contexts[idx]:
                negative_contexts.append(neg_ctx)
        return negative_contexts

class SentenceEmbeddingModel(nn.Module):
    def __init__(self, model_name, pooling_strategy='cls'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.pooling_strategy == 'cls':
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        elif self.pooling_strategy == 'mean':
            # Mean pooling of token embeddings
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return embeddings

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)

class EmbeddingTrainer:
    def __init__(self, model_name, device, pooling_strategy='cls', embedding_dim=768, n_trees=10):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SentenceEmbeddingModel(model_name, pooling_strategy).to(device)
        self.loss_fn = TripletLoss(margin=1.0).to(device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.scheduler = None  # Will be initialized in the train method
        self.best_val_loss = float('inf')

    def train(self, train_dataloader, val_dataloader, 
              num_epochs=3, 
              learning_rate=2e-5,
              warmup_steps=0,
              logging_steps=100,
              rank=0):
        # Initialize scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        for epoch in range(num_epochs):
            if rank == 0:
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            self.model.train()
            epoch_train_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", disable=(rank !=0))
    
            for step, batch in enumerate(progress_bar):
                anchor_input_ids = batch['anchor_input_ids'].to(self.device)
                anchor_attention_mask = batch['anchor_attention_mask'].to(self.device)
                positive_input_ids = batch['positive_input_ids'].to(self.device)
                positive_attention_mask = batch['positive_attention_mask'].to(self.device)
                negative_input_ids = batch['negative_input_ids'].to(self.device)  # Shape: (batch_size, num_negatives, max_length)
                negative_attention_mask = batch['negative_attention_mask'].to(self.device)  # Shape: (batch_size, num_negatives, max_length)
    
                batch_size, num_negatives, max_length = negative_input_ids.size()
                negative_input_ids = negative_input_ids.view(batch_size * num_negatives, max_length)
                negative_attention_mask = negative_attention_mask.view(batch_size * num_negatives, max_length)
    
                # Forward pass
                anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)  # Shape: (batch_size, embedding_dim)
                positive_embeddings = self.model(positive_input_ids, positive_attention_mask)  # Shape: (batch_size, embedding_dim)
                negative_embeddings = self.model(negative_input_ids, negative_attention_mask)  # Shape: (batch_size * num_negatives, embedding_dim)
                negative_embeddings = negative_embeddings.view(batch_size, num_negatives, -1)  # Shape: (batch_size, num_negatives, embedding_dim)
    
                # Compute loss for each negative sample and take the mean
                loss = 0
                for i in range(num_negatives):
                    loss += self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings[:, i, :])
                loss = loss / num_negatives
                epoch_train_loss += loss.item()
    
                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
    
                # Update progress bar
                if rank == 0:
                    progress_bar.set_postfix({'training_loss': f"{loss.item():.4f}"})
    
                # Log metrics
                if rank == 0 and step % logging_steps == 0:
                    wandb.log({
                        'epoch': epoch + 1,
                        'step': step,
                        'train_loss': loss.item(),
                        'learning_rate': self.scheduler.get_last_lr()[0],
                    })
    
            avg_train_loss = epoch_train_loss / len(train_dataloader)
            if rank == 0:
                logger.info(f"Average training loss: {avg_train_loss:.4f}")
    
            # Validation
            val_loss = self.evaluate(val_dataloader, rank)
            if rank == 0:
                logger.info(f"Validation loss: {val_loss:.4f}")
    
            # Save best model
            if val_loss < self.best_val_loss and rank == 0:
                self.best_val_loss = val_loss
                self.save_model("best_model")
                logger.info("Best model saved.")
    
            # Log epoch metrics
            if rank == 0:
                wandb.log({
                    'epoch': epoch + 1,
                    'avg_train_loss': avg_train_loss,
                    'val_loss': val_loss,
                })

    def evaluate(self, dataloader, rank=0):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating", disable=(rank !=0))
            for batch in progress_bar:
                anchor_input_ids = batch['anchor_input_ids'].to(self.device)
                anchor_attention_mask = batch['anchor_attention_mask'].to(self.device)
                positive_input_ids = batch['positive_input_ids'].to(self.device)
                positive_attention_mask = batch['positive_attention_mask'].to(self.device)
                negative_input_ids = batch['negative_input_ids'].to(self.device)  # Shape: (batch_size, num_negatives, max_length)
                negative_attention_mask = batch['negative_attention_mask'].to(self.device)  # Shape: (batch_size, num_negatives, max_length)
    
                batch_size, num_negatives, max_length = negative_input_ids.size()
                negative_input_ids = negative_input_ids.view(batch_size * num_negatives, max_length)
                negative_attention_mask = negative_attention_mask.view(batch_size * num_negatives, max_length)
    
                # Forward pass
                anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)  # Shape: (batch_size, embedding_dim)
                positive_embeddings = self.model(positive_input_ids, positive_attention_mask)  # Shape: (batch_size, embedding_dim)
                negative_embeddings = self.model(negative_input_ids, negative_attention_mask)  # Shape: (batch_size * num_negatives, embedding_dim)
                negative_embeddings = negative_embeddings.view(batch_size, num_negatives, -1)  # Shape: (batch_size, num_negatives, embedding_dim)
    
                # Compute loss for each negative sample and take the mean
                loss = 0
                for i in range(num_negatives):
                    loss += self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings[:, i, :])
                loss = loss / num_negatives
                total_loss += loss.item()
    
        avg_val_loss = total_loss / len(dataloader)
        return avg_val_loss

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.module.encoder.save_pretrained(output_dir)  # .module for DDP
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

def setup(rank, world_size):
    """
    Initialize the process group for DDP.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # You can choose any free port

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    torch.distributed.destroy_process_group()

def main_worker(rank, world_size, args):
    """
    The main worker function to be launched on each process.
    """
    setup(rank, world_size)

    # Set the device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Initialize wandb only on the main process
    if rank == 0:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key  # Pass via arguments
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)

    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Load and preprocess your data
    df = pd.read_csv(args.data_path)

    # Assuming your DataFrame has 'questionID', 'question', 'contextID', 'context' columns
    questions = df['question'].tolist()
    positive_contexts = df['context'].tolist()
    all_contexts = df['context'].unique().tolist()

    # Split data
    train_questions, val_questions, train_positive_contexts, val_positive_contexts = train_test_split(
        questions,
        positive_contexts,
        test_size=args.train_test_split,
        random_state=args.random_seed
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create datasets
    train_dataset = TripletDataset(
        questions=train_questions,
        positive_contexts=train_positive_contexts,
        all_contexts=all_contexts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_negatives=args.num_negatives,
        random_seed=args.random_seed
    )

    val_dataset = TripletDataset(
        questions=val_questions,
        positive_contexts=val_positive_contexts,
        all_contexts=all_contexts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_negatives=args.num_negatives,
        random_seed=args.random_seed
    )

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize trainer
    trainer = EmbeddingTrainer(
        model_name=args.model_name,
        device=device,
        pooling_strategy=args.pooling_strategy,
        embedding_dim=args.embedding_dim,
        n_trees=args.n_trees
    )

    # Wrap the model with DDP
    trainer.model = DDP(trainer.model, device_ids=[rank])

    # Train model
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        rank=rank
    )

    # Cleanup
    cleanup()

def parse_args():
    parser = argparse.ArgumentParser(description="DDP Training for Sentence Embeddings with Triplet Loss")
    parser.add_argument('--model_name', type=str, default='hiieu/halong_embedding', help='Pretrained model name')
    parser.add_argument('--batch_size', type=int, default=16, help='Per-GPU batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps')
    parser.add_argument('--train_test_split', type=float, default=0.2, help='Train-validation split ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_negatives', type=int, default=1, help='Number of negative samples per triplet')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data CSV')
    parser.add_argument('--pooling_strategy', type=str, default='cls', choices=['cls', 'mean'], help='Pooling strategy for embeddings')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Dimension of embeddings')
    parser.add_argument('--n_trees', type=int, default=10, help='Number of trees for Annoy index (if used)')
    parser.add_argument('--wandb_project', type=str, default='sentence-embeddings-training', help='WandB project name')
    parser.add_argument('--wandb_api_key', type=str, required=True, help='WandB API Key')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--logging_steps', type=int, default=100, help='Number of logging steps')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No GPUs available for training.")

    # Launch processes
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

