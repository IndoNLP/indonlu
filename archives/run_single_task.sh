# scratch
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint scratch --step_size 1 --gamma 0.9 --experiment_name scratch_b$4_step1_gamma0.9_lr1e-4_early$3_layer2_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 2 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint scratch --step_size 1 --gamma 0.9 --experiment_name scratch_b$4_step1_gamma0.9_lr1e-4_early$3_layer4_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 4 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint scratch --step_size 1 --gamma 0.9 --experiment_name scratch_b$4_step1_gamma0.9_lr1e-4_early$3_layer6_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 6 --force

# fasttext-cc-id-300-no-oov-uncased
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint fasttext-cc-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-cc-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer2_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 2 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint fasttext-cc-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-cc-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer4_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 4 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint fasttext-cc-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-cc-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer6_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 6 --force

# fasttext-4B-id-300-no-oov-uncased
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint fasttext-4B-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-4B-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer2_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 2 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint fasttext-4B-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-4B-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer4_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 4 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint fasttext-4B-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-4B-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer6_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 6 --force

# mbert
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint bert-base-multilingual-uncased --step_size 1 --gamma 0.9 --experiment_name bert-base-multilingual-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force

# xlm-roberta
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-roberta-base --step_size 1 --gamma 0.9 --experiment_name  xlm-roberta-base_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-roberta-large --step_size 1 --gamma 0.9 --experiment_name  xlm-roberta-large_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 24 --force

# xlm-mlm
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-mlm-100-1280 --step_size 1 --gamma 0.9 --experiment_name xlm-mlm-100-1280_b$4_step1_gamma0.9_lr1e-5_early$3_layer16_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 16 --force

# indobert-base-p1
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint indobenchmark/indobert-base-p1 --step_size 1 --gamma 0.9 --experiment_name indobert-base-p1-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# indobert-base-p2
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint indobenchmark/indobert-base-p2 --step_size 1 --gamma 0.9 --experiment_name indobert-base-p2-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# indobert-large-p1
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint indobenchmark/indobert-large-p1 --step_size 1 --gamma 0.9 --experiment_name indobert-large-p1-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# indobert-large-p2
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint indobenchmark/indobert-large-p2 --step_size 1 --gamma 0.9 --experiment_name indobert-large-p2-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# indobert-lite-base-p1
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint indobenchmark/indobert-lite-base-p1 --step_size 1 --gamma 0.9 --experiment_name indobert-lite-base-p1-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# indobert-lite-base-p2
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint indobenchmark/indobert-lite-base-p2 --step_size 1 --gamma 0.9 --experiment_name indobert-lite-base-p2-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# indobert-lite-large-p1
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint indobenchmark/indobert-lite-large-p1 --step_size 1 --gamma 0.9 --experiment_name indobert-lite-large-p1-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# indobert-lite-large-p2
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint indobenchmark/indobert-lite-large-p2 --step_size 1 --gamma 0.9 --experiment_name indobert-lite-large-p2-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force