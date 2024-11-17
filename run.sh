python train_ddp.py \
    --model_name hiieu/halong_embedding \
    --batch_size 64 \
    --max_length 128 \
    --num_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 0 \
    --train_test_split 0.2 \
    --random_seed 42 \
    --num_negatives 1 \
    --data_path /kaggle/input/bkai-ai-track2-legal-document-retrieval/Legal Document Retrieval/train.csv \
    --pooling_strategy cls \
    --embedding_dim 768 \
    --n_trees 10 \
    --wandb_project sentence-embeddings-training \
    --wandb_api_key 5da9582317ad777fa3c5c8afd54990c3c8ca4187 \
    --num_workers 4

