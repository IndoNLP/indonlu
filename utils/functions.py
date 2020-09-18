from argparse import ArgumentParser
from transformers import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification, AlbertModel
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertForPreTraining, BertModel
from transformers import XLMConfig, XLMTokenizer, XLMForSequenceClassification, XLMForTokenClassification, XLMModel
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaModel
from modules.word_classification import AlbertForWordClassification, BertForWordClassification, XLMForWordClassification, XLMRobertaForWordClassification
from modules.multi_label_classification import AlbertForMultiLabelClassification, BertForMultiLabelClassification, XLMForMultiLabelClassification, XLMRobertaForMultiLabelClassification
import json
import numpy as np
import torch

class WordSplitTokenizer():
    def tokenize(self, string):
        return string.split()
    
class SimpleTokenizer():
    def __init__(self, vocab, word_tokenizer, lower=True):
        self.vocab = vocab
        self.lower = lower
        idx = len(self.vocab.keys())
        self.vocab["<bos>"] = idx+0
        self.vocab["<|endoftext|>"] = idx+1
        self.vocab["<speaker1>"] = idx+2
        self.vocab["<speaker2>"] = idx+3
        self.vocab["<pad>"] = idx+4
        self.vocab["<cls>"] = idx+5
        self.vocab["<sep>"] = idx+6

        self.inverted_vocab = {int(v):k for k,v in self.vocab.items()}
        assert len(self.vocab.keys()) == len(self.inverted_vocab.keys())
        
        # Define word tokenizer
        self.tokenizer = word_tokenizer
        
        # Add special token attribute
        self.cls_token_id = self.vocab["<cls>"]
        self.sep_token_id = self.vocab["<sep>"]   

    def __len__(self):
        return len(self.vocab.keys())+1

    def convert_tokens_to_ids(self,tokens):
        if(type(tokens)==list):
            return [self.vocab[tok] for tok in tokens]
        else:
            return self.vocab[tokens]

    def encode(self,text,text_pair=None,add_special_tokens=False):
        if self.lower:
            text = text.lower()
            text_pair = text_pair.lower() if text_pair else None

        if not add_special_tokens:
            tokens = [self.vocab[tok] for tok in self.tokenizer.tokenize(text)]
            if text_pair:
                tokens += [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)]
        else:
            tokens = [self.vocab["<cls>"]] + [self.vocab[tok] for tok in self.tokenizer.tokenize(text)] + [self.vocab["<sep>"]]
            if text_pair:
                tokens += [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)] + [self.vocab["<sep>"]]
        return tokens     
    
    def encode_plus(self,text,text_pair=None,add_special_tokens=False, return_token_type_ids=False):
        if self.lower:
            text = text.lower()
            text_pair = text_pair.lower() if text_pair else None
        
        if not add_special_tokens:
            tokens = [self.vocab[tok] for tok in self.tokenizer.tokenize(text)]
            if text_pair:
                tokens_pair = [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)]
                token_type_ids = len(tokens) * [0] + len(tokens_pair) * [1]
                tokens += tokens_pair
        else:
            tokens = [self.vocab["<cls>"]] + [self.vocab[tok] for tok in self.tokenizer.tokenize(text)] + [self.vocab["<sep>"]]
            if text_pair:
                tokens_pair = [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)] + [self.vocab["<sep>"]]
                token_type_ids = (len(tokens) * [0]) + (len(tokens_pair) * [1])
                tokens += tokens_pair
        
        encoded_inputs = {}
        encoded_inputs['input_ids'] = tokens
        if return_token_type_ids:
            encoded_inputs['token_type_ids'] = token_type_ids
        return encoded_inputs

    def decode(self,index,skip_special_tokens=True):
        return " ".join([self.inverted_vocab[ind] for ind in index])

    def save_pretrained(self, save_dir): 
        with open(save_dir+'/vocab.json', 'w') as fp:
            json.dump(self.vocab, fp, indent=4)

def gen_embeddings(vocab_list, emb_path, emb_dim=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = None
    count, pre_trained = 0, 0
    vocab_map = {}
    for i in range(len(vocab_list)):
        vocab_map[vocab_list[i]] = i

    found_word_map = {}

    print('Loading embedding file: %s' % emb_path)
    for line in open(emb_path).readlines():
        sp = line.split()
        count += 1
        if count == 1 and emb_dim is None:
            # header <num_vocab, emb_dim>
            emb_dim = int(sp[1])
            embeddings = np.random.rand(len(vocab_list), emb_dim)
            print('Embeddings: %d x %d' % (len(vocab_list), emb_dim))
        else:
            if count == 1:
                embeddings = np.random.rand(len(vocab_list), emb_dim)
                print('Embeddings: %d x %d' % (len(vocab_list), emb_dim))
                continue

            if(len(sp) == emb_dim + 1): 
                if sp[0] in vocab_map:
                    found_word_map[sp[0]] = True
                    embeddings[vocab_map[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print("Error:", sp[0], len(sp))
    pre_trained = len(found_word_map)
    print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / len(vocab_list)))
    return embeddings

def load_vocab(path):
    vocab_list = []
    with open(path, "r") as f:
        for word in f:
            vocab_list.append(word.replace('\n',''))

    vocab_map = {}
    for i in range(len(vocab_list)):
        vocab_map[vocab_list[i]] = i
        
    return vocab_list, vocab_map

def get_model_class(model_type, task):
    if 'babert-lite' in model_type:
        base_cls = AlbertModel
        if 'sequence_classification' == task:
            pred_cls = AlbertForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = AlbertForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = AlbertForMultiLabelClassification     
    elif 'xlm-mlm' in model_type:
        base_cls = XLMModel
        if 'sequence_classification' == task:
            pred_cls = XLMForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = XLMForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = XLMForMultiLabelClassification
    elif 'xlm-roberta' in model_type:
        base_cls = XLMRobertaModel
        if 'sequence_classification' == task:
            pred_cls = XLMRobertaForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = XLMRobertaForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = XLMRobertaForMultiLabelClassification
    else: # 'babert', 'bert-base-multilingual', 'word2vec', 'fasttext', 'scratch'
        base_cls = BertModel
        if 'sequence_classification' == task:
            pred_cls = BertForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = BertForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = BertForMultiLabelClassification
    return base_cls, pred_cls

def load_word_embedding_model(model_type, task, vocab_path, word_tokenizer_class, emb_path, num_labels, lower=True):
    # Load config
    config = BertConfig.from_pretrained('bert-base-uncased') 

    # Init word tokenizer
    word_tokenizer = word_tokenizer_class()
    
    # Load vocab
    _, vocab_map = load_vocab(vocab_path)
    tokenizer = SimpleTokenizer(vocab_map, word_tokenizer, lower=lower)
    vocab_list = list(tokenizer.vocab.keys())

    # Adjust config
    if type(num_labels) == list:
        config.num_labels = max(num_labels)
        config.num_labels_list = num_labels
    else:
        config.num_labels = num_labels
    config.num_hidden_layers = num_labels
    
    if 'word2vec' in model_type:
        embeddings = gen_embeddings(vocab_list, emb_path)
        config.hidden_size = 400
        config.num_attention_heads = 8                                                        
    else: # 'fasttext'
        embeddings = gen_embeddings(vocab_list, emb_path, emb_dim=300)
        config.hidden_size = 300
        config.num_attention_heads = 10  
    config.vocab_size = len(embeddings)

    # Instantiate model
    if 'sequence_classification' == task:
        model = BertForSequenceClassification(config)
        model.bert.embeddings.word_embeddings.weight.data.copy_(torch.FloatTensor(embeddings))
    elif 'token_classification' == task:
        model = BertForWordClassification(config)
        model.bert.embeddings.word_embeddings.weight.data.copy_(torch.FloatTensor(embeddings))
    elif 'multi_label_classification' == task:
        model = BertForMultiLabelClassification(config)
        model.bert.embeddings.word_embeddings.weight.data.copy_(torch.FloatTensor(embeddings))        
    return model, tokenizer

def load_eval_model(args):
    vocab_path = f'./{args["model_dir"]}/{args["dataset"]}/{args["experiment_name"]}/vocab.txt'
    config_path = f'./{args["model_dir"]}/{args["dataset"]}/{args["experiment_name"]}/config.json'
    model_path = f'./{args["model_dir"]}/{args["dataset"]}/{args["experiment_name"]}/best_model_0.th'
    
    # Load for word2vec and fasttext
    if 'word2vec' in args['model_type'] or 'fasttext' in args['model_type']:
        emb_path = args['embedding_path'][args['model_type']]
        model, tokenizer = load_word_embedding_model(
            args['model_type'], args['task'], vocab_path, 
            args['word_tokenizer_class'], emb_path, args['num_labels'], lower=args['lower']
        )
        return model, tokenizer
        
    # Load config & tokenizer
    if 'albert' in args['model_type']:
        config = AlbertConfig.from_json_file(config_path)
        tokenizer = BertTokenizer(vocab_path)
    elif 'babert' in args['model_type']:
        config = BertConfig.from_json_file(config_path)
        tokenizer = BertTokenizer(vocab_path)
    elif 'scratch' in args['model_type']:
        config = BertConfig.from_pretrained('bert-base-uncased') 
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif 'bert-base-multilingual' in args['model_type']:
        config = BertConfig.from_pretrained(args['model_type'])
        tokenizer = BertTokenizer.from_pretrained(args['model_type'])
    elif 'xlm-mlm-100-1280' in args['model_type']:
        config = XLMConfig.from_pretrained(args['model_type'])
        tokenizer = XLMTokenizer.from_pretrained(args['model_type'])
    elif 'xlm-roberta' in args['model_type']:
        config = XLMRobertaConfig.from_pretrained(args['model_type'])
        tokenizer = XLMRobertaTokenizer.from_pretrained(args['model_type'])
    else:
        raise ValueError('Invalid `model_type` argument values')
    
    # Get model class
    base_cls, pred_cls = get_model_class(args['model_type'], args['task'])
        
    # Adjust config
    if type(args['num_labels']) == list:
        config.num_labels = max(args['num_labels'])
        config.num_labels_list = args['num_labels']
    else:
        config.num_labels = args['num_labels']    
        
    # Instantiate model
    model = pred_cls(config=config)
    base_model = base_cls.from_pretrained(model_path, from_tf=False, config=config)
    
    # Plug pretrained base model to classification model
    if 'bert' in model.__dir__():
        model.bert = base_model
    elif 'albert' in model.__dir__():
        model.albert = base_model
    elif 'roberta' in model.__dir__():
        model.roberta = base_model
    elif 'transformer' in model.__dir__():
        model.transformer = base_model
    else:
        ValueError('Model attribute not found, is there any change in the `transformers` library?')    
                                                        
    return model, tokenizer

def load_model(args):
    if 'albert-large-wwmlm-512' == args['model_checkpoint']:
        vocab_path="../embeddings/albert-large-wwmlm-512/albert_large_model_bpe_wwmlm_512_vocab_uncased_30000.txt"
        tokenizer = BertTokenizer(vocab_path)
        config = AlbertConfig.from_json_file("../embeddings/albert-large-wwmlm-512/albert_large_model_bpe_wwmlm_512_albert_large_config.json")
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = AlbertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = AlbertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = AlbertForMultiLabelClassification(config)

        # Plug pretrained bert model
        albert_model = AlbertModel.from_pretrained("../embeddings/albert-large-wwmlm-512/albert_large_model_bpe_wwmlm_512_pytorch_albert_large_512_629k.bin", from_tf=False, config=config)
        model.albert = albert_model
    elif 'albert-base-wwmlm-512' == args['model_checkpoint']:
        vocab_path="../embeddings/albert-base-wwmlm-512/albert_base_model_bpe_wwmlm_512_vocab_uncased_30000.txt"
        config_path = "../embeddings/albert-base-wwmlm-512/albert_base_model_bpe_wwmlm_512_albert_base_config.json"
        
        tokenizer = BertTokenizer(vocab_path)
        config = AlbertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = AlbertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = AlbertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = AlbertForMultiLabelClassification(config)

        # Plug pretrained bert model
        albert_model = AlbertModel.from_pretrained("../embeddings/albert-base-wwmlm-512/albert_base_model_bpe_wwmlm_512_pytorch_model_albert_base_162k.bin", from_tf=False, config=config)
        model.albert = albert_model

    elif 'albert-large-wwmlm-128' == args['model_checkpoint']:
        vocab_path="../embeddings/albert-large-wwmlm-128/albert_large_model_bpe_wwmlm_128_vocab_uncased_30000.txt"
        config_path = "../embeddings/albert-large-wwmlm-128/albert_large_model_bpe_wwmlm_128_albert_large_config.json"

        tokenizer = BertTokenizer(vocab_path)
        config = AlbertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = AlbertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = AlbertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = AlbertForMultiLabelClassification(config)

        # Plug pretrained bert model
        albert_model = AlbertModel.from_pretrained("../embeddings/albert-large-wwmlm-128/albert_large_model_bpe_wwmlm_128_pytorch_albert_large_128_500k.bin", from_tf=False, config=config)
        model.albert = albert_model

    elif 'babert-bpe-mlm-large-512' == args['model_checkpoint']:
        # babert_bpe
        # Prepare config & tokenizer
        vocab_path="../embeddings/babert-bpe-mlm-large-512/babert_model_bpe_mlm_uncased_large_512_dup10-5_vocab_uncased_30522.txt"
        config_path = "../embeddings/babert-bpe-mlm-large-512/babert_model_bpe_mlm_uncased_large_512_dup10-5_bert_large_config.json"
        
        tokenizer = BertTokenizer(vocab_path)
        config = BertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = BertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = BertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = BertForMultiLabelClassification(config)

        # Plug pretrained bert model
        bert_model = BertForPreTraining.from_pretrained("../embeddings/babert-bpe-mlm-large-512/babert_model_bpe_mlm_uncased_large_512_dup10-5_pytorch_babert_uncased_large_512_dup10-5_1120k.bin", config=config)
        model.bert = bert_model.bert

    elif 'albert-base-uncased-112500' == args['model_checkpoint']:
        vocab_path="../embeddings/albert-base-uncased-112500/vocab.txt"
        config_path="../embeddings/albert-base-uncased-112500/bert_config.json"
        
        tokenizer = BertTokenizer(vocab_path)
        config = AlbertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = AlbertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = AlbertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = AlbertForMultiLabelClassification(config)

        # Plug pretrained bert model
        albert_model = AlbertModel.from_pretrained("../embeddings/albert-base-uncased-112500/albert_base_uncased_112500.bin", from_tf=False, config=config)
        model.albert = albert_model
    
    elif 'albert-base-uncased-96000' == args['model_checkpoint']:
        vocab_path="../embeddings/albert-base-uncased-96000/vocab.txt"
        config_path="../embeddings/albert-base-uncased-96000/bert_config.json"
        
        tokenizer = BertTokenizer(vocab_path)
        config = AlbertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = AlbertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = AlbertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = AlbertForMultiLabelClassification(config)

        # Plug pretrained bert model
        albert_model = AlbertModel.from_pretrained("../embeddings/albert-base-uncased-96000/albert_base_uncased_96000.bin", from_tf=False, config=config)
        model.albert = albert_model

    elif 'albert-base-uncased-191k' == args['model_checkpoint']:
        vocab_path="../embeddings/albert-base-uncased-191k/pytorch_models_albert_base_uncased_191500_vocab_uncased_30000.txt"
        config_path="../embeddings/albert-base-uncased-191k/pytorch_models_albert_base_uncased_191500_albert_base_config.json"
        
        tokenizer = BertTokenizer(vocab_path)
        config = AlbertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = AlbertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = AlbertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = AlbertForMultiLabelClassification(config)

        # Plug pretrained bert model
        albert_model = AlbertModel.from_pretrained("../embeddings/albert-base-uncased-191k/pytorch_models_albert_base_uncased_191500_pytorch_model_albert_base_191k.bin", from_tf=False, config=config)
        model.albert = albert_model
        
    elif 'babert-opensubtitle' == args['model_checkpoint']:
        # babert-opensubtitle
        # Prepare config & tokenizer
        vocab_path="../embeddings/babert-opensubtitle/vocab.txt"
        config_path="../embeddings/babert-opensubtitle/bert_config.json"
        
        tokenizer = BertTokenizer(vocab_path)
        config = BertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = BertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = BertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = BertForMultiLabelClassification(config)

        # Plug pretrained bert model
        bert_model = BertForPreTraining.from_pretrained("../embeddings/babert-opensubtitle/model.ckpt-1000000.index", from_tf=True, config=config)
        model.bert = bert_model.bert

    elif 'babert-bpe-mlm-large-uncased-1100k' == args['model_checkpoint']:
        # babert_bpe
        # Prepare config & tokenizer
        vocab_path="../embeddings/babert-bpe-mlm-large-uncased-1100k/pytorch_models_babert_uncased_large_1100k_vocab_uncased_30522.txt"
        config_path="../embeddings/babert-bpe-mlm-large-uncased-1100k/pytorch_models_babert_uncased_large_1100k_bert_config.json"

        tokenizer = BertTokenizer(vocab_path)
        config = BertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = BertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = BertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = BertForMultiLabelClassification(config)

        # Plug pretrained bert model
        bert_model = BertForPreTraining.from_pretrained("../embeddings/babert-bpe-mlm-large-uncased-1100k/pytorch_models_babert_uncased_large_1100k_pytorch_model_babert_large_1100k.bin", config=config)
        model.bert = bert_model.bert

    elif 'babert-bpe-mlm-large-uncased-1m' == args['model_checkpoint']:
        # babert_bpe
        # Prepare config & tokenizer
        vocab_path="../embeddings/babert-bpe-mlm-large-uncased-1m/pytorch_models_babert_uncased_large_1mil_vocab_uncased_30522.txt"
        config_path="../embeddings/babert-bpe-mlm-large-uncased-1m/pytorch_models_babert_uncased_large_1mil_bert_config.json"
        
        tokenizer = BertTokenizer(vocab_path)
        config = BertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = BertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = BertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = BertForMultiLabelClassification(config)

        # Plug pretrained bert model
        bert_model = BertForPreTraining.from_pretrained("../embeddings/babert-bpe-mlm-large-uncased-1m/pytorch_models_babert_uncased_large_1mil_pytorch_model_babert_large_1mil.bin", config=config)
        model.bert = bert_model.bert

    elif 'babert-base-512' == args['model_checkpoint']:
        # babert_bpe
        # Prepare config & tokenizer
        vocab_path="../embeddings/babert-base-512/pytorch_models_babert_base_512_vocab_uncased_30522.txt"
        config_path="../embeddings/babert-base-512/pytorch_models_babert_base_512_bert_config.json"
        
        tokenizer = BertTokenizer(vocab_path)
        config = BertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = BertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = BertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = BertForMultiLabelClassification(config)

        # Plug pretrained bert model
        bert_model = BertForPreTraining.from_pretrained("../embeddings/babert-base-512/pytorch_models_babert_base_512_pytorch_model_babert_base_uncased_512.bin", config=config)
        model.bert = bert_model.bert

    elif 'babert-bpe-mlm-large-uncased' == args['model_checkpoint']:
        # babert_bpe
        # Prepare config & tokenizer
        vocab_path="../embeddings/babert-bpe-mlm-large-uncased/pytorch_models_babert_uncased_large_vocab_uncased_30522.txt"
        config_path="../embeddings/babert-bpe-mlm-large-uncased/pytorch_models_babert_uncased_large_bert_config.json"
        
        tokenizer = BertTokenizer(vocab_path)
        config = BertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = BertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = BertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = BertForMultiLabelClassification(config)

        # Plug pretrained bert model
        bert_model = BertForPreTraining.from_pretrained("../embeddings/babert-bpe-mlm-large-uncased/pytorch_models_babert_uncased_large_pytorch_model_babert_large_778500.bin", config=config)
        model.bert = bert_model.bert

    elif 'babert-bpe-mlm-uncased-128-dup10-5' == args['model_checkpoint']:
        # babert_bpe_wwmlm
        # Prepare config & tokenizer
        vocab_path="../embeddings/babert-bpe-mlm-uncased-128-dup10-5/vocab.txt"
        config_path="../embeddings/babert-bpe-mlm-uncased-128-dup10-5/bert_config.json"
        
        tokenizer = BertTokenizer(vocab_path)
        config = BertConfig.from_json_file(config_path)
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = BertForSequenceClassification(config)
        elif 'token_classification' == args['task']:
            model = BertForWordClassification(config)
        elif 'multi_label_classification' == args['task']:
            model = BertForMultiLabelClassification(config)

        # Plug pretrained bert model
        bert_model = BertForPreTraining.from_pretrained("../embeddings/babert-bpe-mlm-uncased-128-dup10-5/pytorch_model.bin", config=config)
        model.bert = bert_model.bert
        
    elif 'bert-base-multilingual' in args['model_checkpoint']:
        # bert-base-multilingual-uncased or bert-base-multilingual-cased
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = BertTokenizer.from_pretrained(args['model_checkpoint'])
        config = BertConfig.from_pretrained(args['model_checkpoint'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        
        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = BertForSequenceClassification.from_pretrained(args['model_checkpoint'], config=config)
        elif 'token_classification' == args['task']:
            model = BertForWordClassification.from_pretrained(args['model_checkpoint'], config=config)
        elif 'multi_label_classification' == args['task']:
            model = BertForMultiLabelClassification.from_pretrained(args['model_checkpoint'], config=config)
            
    elif 'xlm-mlm' in args['model_checkpoint']:
        # xlm-mlm-100-1280
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = XLMTokenizer.from_pretrained(args['model_checkpoint'])            
        config = XLMConfig.from_pretrained(args['model_checkpoint'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = XLMForSequenceClassification.from_pretrained(args['model_checkpoint'], config=config)
        elif 'token_classification' == args['task']:
            model = XLMForWordClassification.from_pretrained(args['model_checkpoint'], config=config)
        elif 'multi_label_classification' == args['task']:
            model = XLMForMultiLabelClassification.from_pretrained(args['model_checkpoint'], config=config)
    elif 'xlm-roberta' in args['model_checkpoint']:
        # xlm-roberta-base or xlm-roberta-large
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = XLMRobertaTokenizer.from_pretrained(args['model_checkpoint'])                                                        
        config = XLMRobertaConfig.from_pretrained(args['model_checkpoint'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        
        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = XLMRobertaForSequenceClassification.from_pretrained(args['model_checkpoint'], config=config)
        elif 'token_classification' == args['task']:
            model = XLMRobertaForWordClassification.from_pretrained(args['model_checkpoint'], config=config)
        elif 'multi_label_classification' == args['task']:
            model = XLMRobertaForMultiLabelClassification.from_pretrained(args['model_checkpoint'], config=config)
    elif 'word2vec' in args['model_checkpoint'] or 'fasttext' in args['model_checkpoint']:
        # Prepare config & tokenizer
        vocab_path = args['vocab_path']
        config_path = None
        
        word_tokenizer = args['word_tokenizer_class']()
        emb_path = args['embedding_path'][args['model_checkpoint']]

        _, vocab_map = load_vocab(vocab_path)
        tokenizer = SimpleTokenizer(vocab_map, word_tokenizer, lower=args["lower"])
        vocab_list = list(tokenizer.vocab.keys())

        config = BertConfig.from_pretrained('bert-base-uncased') 
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        config.num_hidden_layers = args["num_layers"]

        if args['model_checkpoint'] == 'word2vec-twitter':
            embeddings = gen_embeddings(vocab_list, emb_path)
            config.hidden_size = 400
            config.num_attention_heads = 8

        if args['model_checkpoint'] == 'fasttext-cc-id' or args['model_checkpoint'] == 'fasttext-cc-id-300-no-oov-uncased' or args['model_checkpoint'] == 'fasttext-4B-id-300-no-oov-uncased':
            embeddings = gen_embeddings(vocab_list, emb_path, emb_dim=300)
            config.hidden_size = 300
            config.num_attention_heads = 10

        config.vocab_size = len(embeddings)

        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = BertForSequenceClassification(config)
            model.bert.embeddings.word_embeddings.weight.data.copy_(torch.FloatTensor(embeddings))
        elif 'token_classification' == args['task']:
            model = BertForWordClassification(config)
            model.bert.embeddings.word_embeddings.weight.data.copy_(torch.FloatTensor(embeddings))
        elif 'multi_label_classification' == args['task']:
            model = BertForMultiLabelClassification(config)
            model.bert.embeddings.word_embeddings.weight.data.copy_(torch.FloatTensor(embeddings))
            
    elif 'scratch' in args['model_checkpoint']: 
        vocab_path, config_path = None, None
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        config = BertConfig.from_pretrained("bert-base-uncased")
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        config.num_hidden_layers = args["num_layers"]
        config.hidden_size = 300
        config.num_attention_heads = 10
        
        if 'sequence_classification' == args['task']:
            model = BertForSequenceClassification(config=config)
        elif 'token_classification' == args['task']:
            model = BertForWordClassification(config=config)
        elif 'multi_label_classification' == args['task']:
            model = BertForMultiLabelClassification(config=config)
    elif 'indobenchmark' in args['model_checkpoint']:
        # indobenchmark models
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = BertTokenizer.from_pretrained(args['model_checkpoint'])
        config = BertConfig.from_pretrained(args['model_checkpoint'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        
        # Instantiate model
        if 'sequence_classification' == args['task']:
            model = BertForSequenceClassification.from_pretrained(args['model_checkpoint'], config=config)
        elif 'token_classification' == args['task']:
            model = BertForWordClassification.from_pretrained(args['model_checkpoint'], config=config)
        elif 'multi_label_classification' == args['task']:
            model = BertForMultiLabelClassification.from_pretrained(args['model_checkpoint'], config=config)
            
    return model, tokenizer, vocab_path, config_path
