# albert-base-uncased-96000
# CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint albert-base-uncased-96000 --step_size 1 --gamma 0.9 --experiment_name albert-base-uncased-96000_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 12 --force

# # albert-base-uncased-96000-spm
# CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint albert-base-uncased-96000-spm --step_size 1 --gamma 0.9 --experiment_name albert-base-uncased-96000-spm_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 12 --force

# # albert-base-uncased-112500-spm
# CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint albert-base-uncased-112500-spm --step_size 1 --gamma 0.9 --experiment_name albert-base-uncased-112500-spm_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 12 --force

# scratch
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint scratch --step_size 1 --gamma 0.9 --experiment_name scratch_b$4_step1_gamma0.9_lr1e-4_early$3_layer2_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 2 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint scratch --step_size 1 --gamma 0.9 --experiment_name scratch_b$4_step1_gamma0.9_lr1e-4_early$3_layer4_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 4 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint scratch --step_size 1 --gamma 0.9 --experiment_name scratch_b$4_step1_gamma0.9_lr1e-4_early$3_layer6_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 6 --force

# fasttext-cc-id-300-no-oov-uncased
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint fasttext-cc-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-cc-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer2_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 2 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint fasttext-cc-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-cc-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer4_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 4 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint fasttext-cc-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-cc-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer6_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 6 --force

# fasttext-4B-id-300-no-oov-uncased
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint fasttext-4B-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-4B-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer2_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 2 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint fasttext-4B-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-4B-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer4_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 4 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint fasttext-4B-id-300-no-oov-uncased --step_size 1 --gamma 0.9 --experiment_name fasttext-4B-id-300-no-oov-uncased_b$4_step1_gamma0.9_lr1e-4_early$3_layer6_lowerTrue --lr 1e-4 --early_stop $3 --dataset $2 --lower --num_layers 6 --force

# babert-base-512
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint babert-base-512 --step_size 1 --gamma 0.9 --experiment_name babert-base-512_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 12 --force

# babert-bpe-mlm-large-512
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint babert-bpe-mlm-large-512 --step_size 1 --gamma 0.9 --experiment_name babert-bpe-mlm-large-512_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# mbert
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint bert-base-multilingual-uncased --step_size 1 --gamma 0.9 --experiment_name bert-base-multilingual-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force

# xlm-roberta
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint xlm-roberta-base --step_size 1 --gamma 0.9 --experiment_name  xlm-roberta-base_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint xlm-roberta-large --step_size 1 --gamma 0.9 --experiment_name  xlm-roberta-large_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 24 --force

# babert-opensubtitle
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint babert-opensubtitle --step_size 1 --gamma 0.9 --experiment_name babert-opensubtitle_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force

# xlm
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint xlm-mlm-100-1280 --step_size 1 --gamma 0.9 --experiment_name xlm-mlm-100-1280_b$4_step1_gamma0.9_lr1e-5_early$3_layer16_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 16 --force

# albert-large-wwmlm-128
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint albert-large-wwmlm-128 --step_size 1 --gamma 0.9 --experiment_name albert-large-wwmlm-128_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# albert-base-wwmlm-512
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint albert-base-wwmlm-512 --step_size 1 --gamma 0.9 --experiment_name albert-base-wwmlm-512_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 12 --force

# albert-large-wwmlm-512
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint albert-large-wwmlm-512 --step_size 1 --gamma 0.9 --experiment_name albert-large-wwmlm-512_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# albert-base-uncased-112500
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint albert-base-uncased-112500 --step_size 1 --gamma 0.9 --experiment_name albert-base-uncased-112500_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 12 --force

# albert-base-uncased-191k
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint albert-base-uncased-191k --step_size 1 --gamma 0.9 --experiment_name albert-base-uncased-191k_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 12 --force

# cartobert
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint cartobert --step_size 1 --gamma 0.9 --experiment_name cartobert_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 12 --force

# babert-bpe-mlm-large-uncased
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint babert-bpe-mlm-large-uncased --step_size 1 --gamma 0.9 --experiment_name babert-bpe-mlm-large-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# babert-bpe-mlm-large-uncased-1m
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint babert-bpe-mlm-large-uncased-1m --step_size 1 --gamma 0.9 --experiment_name babert-bpe-mlm-large-uncased-1m_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# babert-bpe-mlm-large-uncased-1100k
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint babert-bpe-mlm-large-uncased-1100k --step_size 1 --gamma 0.9 --experiment_name babert-bpe-mlm-large-uncased-1100k_b$4_step1_gamma0.9_lr1e-5_early$3_layer24_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 24 --force

# babert-bpe-mlm-uncased-128-dup10-5
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 100 --train_batch_size $4 --model_checkpoint babert-bpe-mlm-uncased-128-dup10-5 --step_size 1 --gamma 0.9 --experiment_name babert-bpe-mlm-uncased-128-dup10-5_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerTrue --lr 1e-5 --early_stop $3 --dataset $2 --lower --num_layers 12 --force
