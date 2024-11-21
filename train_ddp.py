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
import joblib
import multiprocessing as mp
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding_trainer")
class ClassificationDataset(Dataset):
    """
    A custom Dataset class for classification using Cross Entropy Loss.
    Each sample consists of a question, a context, and a label indicating
    whether the context is positive (1) or negative (0) for the question.
    """
    def __init__(self, questions, contexts, labels, tokenizer, max_length=512):
        self.questions = questions
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentenceEmbeddingModel(nn.Module):
    def __init__(self, model_name, pooling_strategy='cls'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.pooling_strategy == 'cls':
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == 'mean':
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        
        # Normalize embeddings
        return F.normalize(embeddings, p=2, dim=1)

class EmbeddingTrainer:
    def __init__(self, model_name, device, pooling_strategy='cls', embedding_dim=768):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SentenceEmbeddingModel(model_name, pooling_strategy).to(device)
        self.loss_fn = nn.CrossEntropyLoss().to(device)  # Changed to CrossEntropyLoss
        self.optimizer = None  # Will be initialized after DDP wrapper
        self.scheduler = None
        self.best_val_loss = float('inf')
        self.classifier = None  # Initialize classifier later

    def train(self, train_dataloader, val_dataloader, 
              num_epochs=3, 
              learning_rate=2e-5,
              warmup_steps=0,
              logging_steps=100,
              rank=0):
              
        # Initialize optimizer after DDP wrapper
        if self.optimizer is None:
            # Include classifier parameters if classifier exists
            if self.classifier:
                self.optimizer = AdamW(list(self.model.parameters()) + list(self.classifier.parameters()), lr=learning_rate)
            else:
                self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
                
        # Initialize scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        for epoch in range(num_epochs):
            train_dataloader.sampler.set_epoch(epoch)  # Important for DDP
            
            if rank == 0:
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            self.model.train()
            if self.classifier:
                self.classifier.train()
            epoch_train_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", disable=(rank != 0))

            for step, batch in enumerate(progress_bar):
                try:
                    # Move all tensors to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    # Forward pass
                    embeddings = self.model(input_ids, attention_mask)

                    # Initialize classifier if not already
                    if self.classifier is None:
                        self.classifier = nn.Linear(embeddings.size(-1), 2).to(self.device)
                        # Wrap classifier with DDP
                        self.classifier = DDP(
                            self.classifier,
                            device_ids=[self.device.index],
                            output_device=self.device.index
                        )
                        # Update optimizer to include classifier parameters
                        self.optimizer = AdamW(list(self.model.parameters()) + list(self.classifier.parameters()), lr=learning_rate)

                    # Compute logits
                    logits = self.classifier(embeddings)  # Shape: (batch_size, 2)

                    # Compute loss
                    loss = self.loss_fn(logits, labels)

                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    epoch_train_loss += loss.item()

                    if rank == 0:
                        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

                        if step % logging_steps == 0:
                            wandb.log({
                                'epoch': epoch + 1,
                                'step': step,
                                'train_loss': loss.item(),
                                'learning_rate': self.scheduler.get_last_lr()[0],
                            })

                except Exception as e:
                    logger.error(f"Error in training step: {str(e)}")
                    continue

            avg_train_loss = epoch_train_loss / len(train_dataloader)
            
            if rank == 0:
                logger.info(f"Average training loss: {avg_train_loss:.4f}")
                
                # Validation
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"Validation loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model("best_model")
                    logger.info("Best model saved.")

                # Log epoch metrics
                wandb.log({
                    'epoch': epoch + 1,
                    'avg_train_loss': avg_train_loss,
                    'val_loss': val_loss,
                })

    @torch.no_grad()
    def evaluate(self, dataloader, rank=0):
        self.model.eval()
        if self.classifier:
            self.classifier.eval()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Evaluating", disable=(rank != 0)):
            try:
                # Move all tensors to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                embeddings = self.model(input_ids, attention_mask)

                # Compute logits
                if self.classifier is None:
                    self.classifier = nn.Linear(embeddings.size(-1), 2).to(self.device)
                    self.classifier = DDP(
                        self.classifier,
                        device_ids=[self.device.index],
                        output_device=self.device.index
                    )
                logits = self.classifier(embeddings)  # Shape: (batch_size, 2)

                # Compute loss
                loss = self.loss_fn(logits, labels)
                
                total_loss += loss.item()

            except Exception as e:
                logger.error(f"Error in evaluation step: {str(e)}")
                continue

        return total_loss / len(dataloader)

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Save the underlying model, not the DDP wrapper
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.encoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        # Save the classifier if it exists
        if self.classifier:
            # Save only the underlying classifier module
            classifier_to_save = self.classifier.module if hasattr(self.classifier, 'module') else self.classifier
            torch.save(classifier_to_save.state_dict(), os.path.join(output_dir, 'classifier.pt'))
        logger.info(f"Model saved to {output_dir}")
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
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def generate_neighbor_samples(q, pos_c, all_contexts, num_negatives):
    samples = [(q, pos_c, 1)]  # Positive sample
    
    pos_index = all_contexts.index(pos_c)
    added_negatives = 0
    
    left_offset = right_offset = 1
    
    while added_negatives < num_negatives:
        # Try left side
        if pos_index - left_offset >= 0 and added_negatives < num_negatives:
            samples.append((q, all_contexts[pos_index - left_offset], 0))
            added_negatives += 1
            left_offset += 1
        
        # Try right side
        if pos_index + right_offset < len(all_contexts) and added_negatives < num_negatives:
            samples.append((q, all_contexts[pos_index + right_offset], 0))
            added_negatives += 1
            right_offset += 1
        
        # Prevent infinite loop
        if left_offset + right_offset > len(all_contexts):
            break
    
    return samples

def generate_samples_parallel(questions, positive_contexts, all_contexts, num_negatives, n_jobs=-1):
    # Use all available cores if n_jobs is -1
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    try:
        # Create a pool of workers
        with mp.Pool(processes=n_jobs) as pool:
            # Prepare arguments for parallel processing
            args = [(q, pos_c, all_contexts, num_negatives) 
                    for q, pos_c in zip(questions, positive_contexts)]
            
            # Use tqdm to show progress
            results = list(tqdm(
                pool.starmap(generate_neighbor_samples, args), 
                total=len(questions), 
                desc="Generating Samples"
            ))
        
        # Flatten results and separate into lists
        questions_list = []
        contexts_list = []
        labels_list = []
        
        for sample_group in results:
            for q, c, label in sample_group:
                questions_list.append(q)
                contexts_list.append(c)
                labels_list.append(label)
        
        # Convert to numpy arrays and release memory
        return (
            np.array(questions_list, dtype=object), 
            np.array(contexts_list, dtype=object), 
            np.array(labels_list, dtype=np.int8)
        )
    
    finally:
        # Explicitly release memory
        import gc
        gc.collect()
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

        # Set seeds for reproducibility
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)  

        # Load data
        df = pd.read_csv(args.data_path)
        questions = df['question'].tolist()
        positive_contexts = df['context'].tolist()
        all_contexts = df['context'].unique().tolist()
        print(f"Data Loaded: {len(questions)} questions, {len(all_contexts)} unique contexts")

        # Determine number of jobs (leave one core free)
        n_jobs = max(1, joblib.cpu_count() - 1)
        print(f"Using {n_jobs} CPU cores for parallel processing")

        # Generate samples in parallel
        questions_list, contexts_list, labels_list = generate_samples_parallel(
            questions, 
            positive_contexts, 
            all_contexts, 
            args.num_negatives,
            n_jobs
        )
        train_questions, val_questions, train_contexts, val_contexts, train_labels, val_labels = train_test_split(
            questions_list,
            contexts_list,
            labels_list,
            test_size=args.train_test_split,
            random_state=args.random_seed
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

