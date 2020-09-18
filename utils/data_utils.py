import numpy as np
import pandas as pd
import string
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def generate_vocab(dataset):
    with open("vocab.txt", "w+") as f_out:
        vocab = {}
        for data in dataset.data:
            words = data['sentence']
            for word in words:
                for w in word.split(" "):
                    vocab[w] = True
        for word in vocab:
            f_out.write(word + "\n")

#####
# Term Extraction Airy
#####
class AspectExtractionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'I-SENTIMENT': 0, 'O': 1, 'I-ASPECT': 2, 'B-SENTIMENT': 3, 'B-ASPECT': 4}
    INDEX2LABEL = {0: 'I-SENTIMENT', 1: 'O', 2: 'I-ASPECT', 3: 'B-SENTIMENT', 4: 'B-ASPECT'}
    NUM_LABELS = 5
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if '\t' in line:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
       
class AspectExtractionDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(AspectExtractionDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)
        
        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]
            
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
    
#####
# Ner Grit + Prosa
#####
class NerGritDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'I-PERSON': 0, 'B-ORGANISATION': 1, 'I-ORGANISATION': 2, 'B-PLACE': 3, 'I-PLACE': 4, 'O': 5, 'B-PERSON': 6}
    INDEX2LABEL = {0: 'I-PERSON', 1: 'B-ORGANISATION', 2: 'I-ORGANISATION', 3: 'B-PLACE', 4: 'I-PLACE', 5: 'O', 6: 'B-PERSON'}
    NUM_LABELS = 7
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data) 

class NerProsaDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'I-PPL': 0, 'B-EVT': 1, 'B-PLC': 2, 'I-IND': 3, 'B-IND': 4, 'B-FNB': 5, 'I-EVT': 6, 'B-PPL': 7, 'I-PLC': 8, 'O': 9, 'I-FNB': 10}
    INDEX2LABEL = {0: 'I-PPL', 1: 'B-EVT', 2: 'B-PLC', 3: 'I-IND', 4: 'B-IND', 5: 'B-FNB', 6: 'I-EVT', 7: 'B-PPL', 8: 'I-PLC', 9: 'O', 10: 'I-FNB'}
    NUM_LABELS = 11
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
        
class NerDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(NerDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)
        
        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
    
#####
# Pos Tag Idn + Prosa
#####
class PosTagIdnDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'B-PR': 0, 'B-CD': 1, 'I-PR': 2, 'B-SYM': 3, 'B-JJ': 4, 'B-DT': 5, 'I-UH': 6, 'I-NND': 7, 'B-SC': 8, 'I-WH': 9, 'I-IN': 10, 'I-NNP': 11, 'I-VB': 12, 'B-IN': 13, 'B-NND': 14, 'I-CD': 15, 'I-JJ': 16, 'I-X': 17, 'B-OD': 18, 'B-RP': 19, 'B-RB': 20, 'B-NNP': 21, 'I-RB': 22, 'I-Z': 23, 'B-CC': 24, 'B-NEG': 25, 'B-VB': 26, 'B-NN': 27, 'B-MD': 28, 'B-UH': 29, 'I-NN': 30, 'B-PRP': 31, 'I-SC': 32, 'B-Z': 33, 'I-PRP': 34, 'I-OD': 35, 'I-SYM': 36, 'B-WH': 37, 'B-FW': 38, 'I-CC': 39, 'B-X': 40}
    INDEX2LABEL = {0: 'B-PR', 1: 'B-CD', 2: 'I-PR', 3: 'B-SYM', 4: 'B-JJ', 5: 'B-DT', 6: 'I-UH', 7: 'I-NND', 8: 'B-SC', 9: 'I-WH', 10: 'I-IN', 11: 'I-NNP', 12: 'I-VB', 13: 'B-IN', 14: 'B-NND', 15: 'I-CD', 16: 'I-JJ', 17: 'I-X', 18: 'B-OD', 19: 'B-RP', 20: 'B-RB', 21: 'B-NNP', 22: 'I-RB', 23: 'I-Z', 24: 'B-CC', 25: 'B-NEG', 26: 'B-VB', 27: 'B-NN', 28: 'B-MD', 29: 'B-UH', 30: 'I-NN', 31: 'B-PRP', 32: 'I-SC', 33: 'B-Z', 34: 'I-PRP', 35: 'I-OD', 36: 'I-SYM', 37: 'B-WH', 38: 'B-FW', 39: 'I-CC', 40: 'B-X'}
    NUM_LABELS = 41
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
    
class PosTagProsaDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'B-PPO': 0, 'B-KUA': 1, 'B-ADV': 2, 'B-PRN': 3, 'B-VBI': 4, 'B-PAR': 5, 'B-VBP': 6, 'B-NNP': 7, 'B-UNS': 8, 'B-VBT': 9, 'B-VBL': 10, 'B-NNO': 11, 'B-ADJ': 12, 'B-PRR': 13, 'B-PRK': 14, 'B-CCN': 15, 'B-$$$': 16, 'B-ADK': 17, 'B-ART': 18, 'B-CSN': 19, 'B-NUM': 20, 'B-SYM': 21, 'B-INT': 22, 'B-NEG': 23, 'B-PRI': 24, 'B-VBE': 25}
    INDEX2LABEL = {0: 'B-PPO', 1: 'B-KUA', 2: 'B-ADV', 3: 'B-PRN', 4: 'B-VBI', 5: 'B-PAR', 6: 'B-VBP', 7: 'B-NNP', 8: 'B-UNS', 9: 'B-VBT', 10: 'B-VBL', 11: 'B-NNO', 12: 'B-ADJ', 13: 'B-PRR', 14: 'B-PRK', 15: 'B-CCN', 16: 'B-$$$', 17: 'B-ADK', 18: 'B-ART', 19: 'B-CSN', 20: 'B-NUM', 21: 'B-SYM', 22: 'B-INT', 23: 'B-NEG', 24: 'B-PRI', 25: 'B-VBE'}
    NUM_LABELS = 26
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
        
class PosTagDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(PosTagDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list

#####
# Emotion Twitter
#####
class EmotionDetectionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'sadness': 0, 'anger': 1, 'love': 2, 'fear': 3, 'happy': 4}
    INDEX2LABEL = {0: 'sadness', 1: 'anger', 2: 'love', 3: 'fear', 4: 'happy'}
    NUM_LABELS = 5
    
    def load_dataset(self, path):
        # Load dataset
        dataset = pd.read_csv(path)
        dataset['label'] = dataset['label'].apply(lambda sen: self.LABEL2INDEX[sen])
        return dataset

    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        tweet, label = self.data.loc[index,'tweet'], self.data.loc[index,'label']        
        subwords = self.tokenizer.encode(tweet, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(label), tweet
    
    def __len__(self):
        return len(self.data)
        
class EmotionDetectionDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(EmotionDetectionDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        label_batch = np.full((batch_size, 1), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            label_batch[i] = label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, label_batch, seq_list

#####
# Entailment UI
#####
class EntailmentDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'NotEntail': 0, 'Entail_or_Paraphrase': 1}
    INDEX2LABEL = {0: 'NotEntail', 1: 'Entail_or_Paraphrase'}
    NUM_LABELS = 2
    
    def load_dataset(self, path):
        df = pd.read_csv(path)
        df['label'] = df['label'].apply(lambda label: self.LABEL2INDEX[label])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        sent_A, sent_B, label = data['sent_A'], data['sent_B'], data['label']
        
        encoded_inputs = self.tokenizer.encode_plus(sent_A, sent_B, add_special_tokens=not self.no_special_token, return_token_type_ids=True)
        subwords, token_type_ids = encoded_inputs["input_ids"], encoded_inputs["token_type_ids"]
        
        return np.array(subwords), np.array(token_type_ids), np.array(label), data['sent_A'] + "|" + data['sent_B']
    
    def __len__(self):
        return len(self.data)
    
        
class EntailmentDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(EntailmentDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        token_type_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        label_batch = np.zeros((batch_size, 1), dtype=np.int64)

        seq_list = []
        
        for i, (subwords, token_type_ids, label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            token_type_batch[i,:len(subwords)] = token_type_ids
            label_batch[i,0] = label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, token_type_batch, label_batch, seq_list

#####
# Document Sentiment Prosa
#####
class DocumentSentimentDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'positive': 0, 'neutral': 1, 'negative': 2}
    INDEX2LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'}
    NUM_LABELS = 3
    
    def load_dataset(self, path): 
        df = pd.read_csv(path, sep='\t', header=None)
        df.columns = ['text','sentiment']
        df['sentiment'] = df['sentiment'].apply(lambda lab: self.LABEL2INDEX[lab])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text, sentiment = data['text'], data['sentiment']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(sentiment), data['text']
    
    def __len__(self):
        return len(self.data)    
        
class DocumentSentimentDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(DocumentSentimentDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        sentiment_batch = np.zeros((batch_size, 1), dtype=np.int64)
        
        seq_list = []
        for i, (subwords, sentiment, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            sentiment_batch[i,0] = sentiment
            
            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, sentiment_batch, seq_list

#####
# Keyword Extraction Prosa
#####
class KeywordExtractionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'O':0, 'B':1, 'I':2}
    INDEX2LABEL = {0:'O', 1:'B', 2:'I'}
    NUM_LABELS = 3
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
        
class KeywordExtractionDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(KeywordExtractionDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []

        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
    
#####
# QA Factoid ITB
#####
class QAFactoidDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'O':0, 'B':1, 'I':2}
    INDEX2LABEL = {0:'O', 1:'B', 2:'I'}
    NUM_LABELS = 3
    
    def load_dataset(self, path):
        # Read file
        dataset = pd.read_csv(path)
        
        # Question and passage are a list of words and seq_label is list of B/I/O
        dataset['question'] = dataset['question'].apply(lambda x: eval(x))
        dataset['passage'] = dataset['passage'].apply(lambda x: eval(x))
        dataset['seq_label'] = dataset['seq_label'].apply(lambda x: [self.LABEL2INDEX[l] for l in eval(x)])

        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        question, passage, seq_label = data['question'],  data['passage'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        token_type_ids = [0]
        
        # Add subwords for question
        for word_idx, word in enumerate(question):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [-1 for i in range(len(subword_list))]
            token_type_ids += [0 for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add intermediate SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        token_type_ids += [0]
        
        # Add subwords
        for word_idx, word in enumerate(passage):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            token_type_ids += [1 for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        token_type_ids += [1]
        
        return np.array(subwords), np.array(token_type_ids), np.array(subword_to_word_indices), np.array(seq_label), ' '.join(question) + "|" + ' '.join(passage)
    
    def __len__(self):
        return len(self.data)
        
class QAFactoidDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(QAFactoidDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[3]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        token_type_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, token_type_ids, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            token_type_batch[i,:len(subwords)] = token_type_ids
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, token_type_batch, subword_to_word_indices_batch, seq_label_batch, seq_list

#####
# ABSA Airy + Prosa
#####
class AspectBasedSentimentAnalysisAiryDataset(Dataset):
    # Static constant variable
    ASPECT_DOMAIN = ['ac', 'air_panas', 'bau', 'general', 'kebersihan', 'linen', 'service', 'sunrise_meal', 'tv', 'wifi']
    LABEL2INDEX = {'neg': 0, 'neut': 1, 'pos': 2, 'neg_pos': 3}
    INDEX2LABEL = {0: 'neg', 1: 'neut', 2: 'pos', 3: 'neg_pos'}
    NUM_LABELS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    NUM_ASPECTS = 10
    
    def load_dataset(self, path):
        df = pd.read_csv(path)
        for aspect in self.ASPECT_DOMAIN:
            df[aspect] = df[aspect].apply(lambda sen: self.LABEL2INDEX[sen])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        sentence, labels = data['review'], [data[aspect] for aspect in self.ASPECT_DOMAIN]
        subwords = self.tokenizer.encode(sentence, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(labels), data['review']
    
    def __len__(self):
        return len(self.data)
    
class AspectBasedSentimentAnalysisProsaDataset(Dataset):
    # Static constant variable
    ASPECT_DOMAIN = ['fuel', 'machine', 'others', 'part', 'price', 'service']
    LABEL2INDEX = {'negative': 0, 'neutral': 1, 'positive': 2}
    INDEX2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}
    NUM_LABELS = [3, 3, 3, 3, 3, 3]
    NUM_ASPECTS = 6
    
    def load_dataset(self, path):
        df = pd.read_csv(path)
        for aspect in self.ASPECT_DOMAIN:
            df[aspect] = df[aspect].apply(lambda sen: self.LABEL2INDEX[sen])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        sentence, labels = data['sentence'], [data[aspect] for aspect in self.ASPECT_DOMAIN]
        subwords = self.tokenizer.encode(sentence, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(labels), data['sentence']
    
    def __len__(self):
        return len(self.data)
    
        
class AspectBasedSentimentAnalysisDataLoader(DataLoader):
    def __init__(self, dataset, max_seq_len=512, *args, **kwargs):
        super(AspectBasedSentimentAnalysisDataLoader, self).__init__(dataset=dataset, *args, **kwargs)
        self.num_aspects = dataset.NUM_ASPECTS
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        label_batch = np.zeros((batch_size, self.num_aspects), dtype=np.int64)

        seq_list = []
        
        for i, (subwords, label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            label_batch[i,:] = label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, label_batch, seq_list

#####
# News Categorization Prosa
#####
class NewsCategorizationDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'permasalahan pada bank besar domestik': 0, 'pertumbuhan ekonomi domestik yang terbatas': 1, 'volatilitas harga komoditas utama dunia': 2, 'frekuensi kenaikan fed fund rate (ffr) yang melebihi ekspektasi': 3, 'perubahan kebijakan dan/atau regulasi pada institusi keuangan': 4, 'isu politik domestik': 5, 'permasalahan pada bank besar international': 6, 'perubahan kebijakan pemerintah yang berkaitan dengan fiskal': 7, 'pertumbuhan ekonomi global yang terbatas': 8, 'kebijakan pemerintah yang bersifat sektoral': 9, 'isu politik dan ekonomi luar negeri': 10, 'kenaikan harga volatile food': 11, 'tidak berisiko': 12, 'pergerakan harga minyak mentah dunia': 13, 'force majeure yang memengaruhi operasional sistem keuangan': 14, 'kenaikan administered price': 15}
    INDEX2LABEL = {0: 'permasalahan pada bank besar domestik', 1: 'pertumbuhan ekonomi domestik yang terbatas', 2: 'volatilitas harga komoditas utama dunia', 3: 'frekuensi kenaikan fed fund rate (ffr) yang melebihi ekspektasi', 4: 'perubahan kebijakan dan/atau regulasi pada institusi keuangan', 5: 'isu politik domestik', 6: 'permasalahan pada bank besar international', 7: 'perubahan kebijakan pemerintah yang berkaitan dengan fiskal', 8: 'pertumbuhan ekonomi global yang terbatas', 9: 'kebijakan pemerintah yang bersifat sektoral', 10: 'isu politik dan ekonomi luar negeri', 11: 'kenaikan harga volatile food', 12: 'tidak berisiko', 13: 'pergerakan harga minyak mentah dunia', 14: 'force majeure yang memengaruhi operasional sistem keuangan', 15: 'kenaikan administered price'}
    NUM_LABELS = 16
    
    def load_dataset(self, path):
        dataset = pd.read_csv(path, sep='\t', header=None)
        dataset.columns = ['text', 'label']
        dataset['label'] = dataset['label'].apply(lambda labels: [self.LABEL2INDEX[label] for label in labels.split(',')])
        return dataset
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text, labels = data['text'], data['label']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(labels), data['text']
    
    def __len__(self):
        return len(self.data)    
        
class NewsCategorizationDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(NewsCategorizationDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        # Trimmed input based on specified max_len
        if self.max_seq_len < max_seq_len:
            max_seq_len = self.max_seq_len
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        labels_batch = np.zeros((batch_size, NUM_LABELS), dtype=np.int64)
        
        seq_list = []
        for i, (subwords, labels) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            for label in labels:
                labels_batch[i,label] = 1

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, labels_batch, seq_list