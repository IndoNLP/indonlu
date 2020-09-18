from copy import deepcopy
import random
import numpy as np
import pandas as pd
import os
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AdamW
from nltk.tokenize import TweetTokenizer

from utils.functions import load_eval_model, WordSplitTokenizer
from utils.args_helper import get_eval_parser, print_opts, append_dataset_args
from utils.metrics import absa_metrics_fn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

###
# modelling functions
###
def get_lr(args, optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

###
# Testing Function
###
def predict(model, data_loader, forward_fn, metrics_fn, i2w, is_test=False):
    model.eval()
    total_loss, total_correct, total_labels = 0, 0, 0

    list_hyp, list_label, list_seq = [], [], []
    pbar = tqdm(iter(data_loader), leave=True, total=len(data_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]
        loss, batch_hyp, batch_label = forward_fn(model, batch_data[:-1], i2w=i2w, device=args['device'])
        
        # Calculate total loss
        test_loss = loss.item()
        total_loss = total_loss + test_loss

        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label
        list_seq += batch_seq
        metrics = metrics_fn(list_hyp, list_label)

        pbar.set_description("TEST LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
    
    return total_loss, metrics, list_hyp, list_label, list_seq

if __name__ == "__main__":
    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True
    
    # Parse args
    args = get_eval_parser()
    args = append_dataset_args(args)

    model_dir = '{}/{}/{}'.format(args["model_dir"],args["dataset"],args['experiment_name'])

    # Set random seed
    set_seed(args['seed'])  # Added here for reproductibility    

    # w2i & i2w
    w2i, i2w = args['dataset_class'].LABEL2INDEX, args['dataset_class'].INDEX2LABEL
    
    if os.path.exists(model_dir + "/best_model_0.th"):
        # load model
        model, tokenizer = load_eval_model(args)
        optimizer = optim.Adam(model.parameters())

        if args['fp16']:
            from apex import amp  # Apex is only required if we use fp16 training
            model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16'])

        if args['device'] == "cuda":
            model = model.cuda()

        print("=========== PREDICTION ===========")

        test_dataset_path = args['test_set_path']
        test_dataset = args['dataset_class'](test_dataset_path, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
        test_loader = args['dataloader_class'](dataset=test_dataset, max_seq_len=args['max_seq_len'], batch_size=args['batch_size'], num_workers=16, shuffle=False)

        _, _, test_hyp, test_label, test_seq = predict(model, test_loader, forward_fn=args['forward_fn'], metrics_fn=args['metrics_fn'], i2w=i2w)

        result_df = pd.DataFrame({
            'seq':test_seq, 
            'hyp': test_hyp, 
            'label': test_label
        })
        print(result_df.head())
        result_df.to_csv(model_dir + "/prediction_result.csv")
    else:
        print(f'Model doesn\'t exist in {model_dir}')