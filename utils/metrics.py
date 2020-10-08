import itertools
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from .conlleval import conll_evaluation

def emotion_detection_metrics_fn(list_hyp, list_label):
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics

def aspect_extraction_metrics_fn(list_hyp, list_label):
    metrics = {}
    acc, pre, rec, f1, tm_pre, tm_rec, tm_f1 = conll_evaluation(list_hyp, list_label)
    metrics["ACC"] = acc
    metrics["F1"] = tm_f1
    metrics["REC"] = tm_rec
    metrics["PRE"] = tm_pre
    return metrics

def ner_metrics_fn(list_hyp, list_label):
    metrics = {}
    acc, pre, rec, f1, tm_pre, tm_rec, tm_f1 = conll_evaluation(list_hyp, list_label)
    metrics["ACC"] = acc
    metrics["F1"] = tm_f1
    metrics["REC"] = tm_rec
    metrics["PRE"] = tm_pre
    return metrics

def pos_tag_metrics_fn(list_hyp, list_label):
    metrics = {}
    acc, pre, rec, f1, tm_pre, tm_rec, tm_f1 = conll_evaluation(list_hyp, list_label)
    metrics["ACC"] = acc
    metrics["F1"] = tm_f1
    metrics["REC"] = tm_rec
    metrics["PRE"] = tm_pre
    return metrics

def entailment_metrics_fn(list_hyp, list_label):
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics

def document_sentiment_metrics_fn(list_hyp, list_label):
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics

def keyword_extraction_metrics_fn(list_hyp, list_label):
    metrics = {}
    acc, pre, rec, f1, tm_pre, tm_rec, tm_f1 = conll_evaluation(list_hyp, list_label)
    metrics["ACC"] = acc
    metrics["F1"] = tm_f1
    metrics["REC"] = tm_rec
    metrics["PRE"] = tm_pre
    return metrics

def qa_factoid_metrics_fn(list_hyp, list_label):
    metrics = {}
    acc, pre, rec, f1, tm_pre, tm_rec, tm_f1 = conll_evaluation(list_hyp, list_label)
    metrics["ACC"] = acc
    metrics["F1"] = tm_f1
    metrics["REC"] = tm_rec
    metrics["PRE"] = tm_pre
    return metrics

def absa_metrics_fn(list_hyp, list_label):
    # hyp and label are both list (multi label), flatten the list
    list_hyp = list(itertools.chain.from_iterable(list_hyp))
    list_label = list(itertools.chain.from_iterable(list_label))
    
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics

def news_categorization_metrics_fn(list_hyp, list_label):
    # hyp and label are both list (multi label), flatten the list
    list_hyp = list(itertools.chain.from_iterable(list_hyp))
    list_label = list(itertools.chain.from_iterable(list_label))
    
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics