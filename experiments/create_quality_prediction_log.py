from typing_model.models.BERT_models import BaseBERTTyper, ConcatenatedContextBERTTyper, OnlyMentionBERTTyper, OnlyContextBERTTyper
from torch.utils.data import DataLoader
import pickle
from torch.nn import Sigmoid 
from collections import defaultdict
import torch
from tqdm import tqdm

model_path = '../typing_experiments/checkpoints/Balanced_Bert_Baseline-v0.ckpt'
dataloader_path = '../typing_experiments/dataloaders/Balanced_Bert_Baseline_ontonotes_dev.pkl'
auxiliary_variables_path =  '../typing_experiments/dataloaders/Bert_Baseline_ontonotes_train_auxiliary_variables.pkl'
weights_path =  '../typing_experiments/datasets_stats/ontonotes_train_weights.pkl'

train_dataset_stats_path = '../typing_experiments/datasets_stats/ontonotes_train.pkl'
dev_dataset_stats_path = '../typing_experiments/datasets_stats/balanced_ontonotes_dev.pkl'

metrics_file = '../typing_experiments/result_logs/quality_prediction/metrics_BalancedBaselineOntonotes.txt'


with open(auxiliary_variables_path, 'rb') as filino:
    id2label, label2id, vocab_len = pickle.load(filino)
with open(weights_path, 'rb') as inp:
    weights = pickle.load(inp)
    # ordered_weights = torch.tensor([weights[id2label[i]] for i in range(len(id2label))])    
    ordered_weights = None    
    model = ConcatenatedContextBERTTyper.load_from_checkpoint(model_path, classes = vocab_len, id2label = id2label, label2id = label2id, weights = ordered_weights)
    model.cuda()
    model.eval()
# model = BaseBERTTyper.load_from_checkpoint(model_path, classes = vocab_len, id2label = id2label, label2id = label2id)

with open(dataloader_path, "rb") as filino:
    dataloader = pickle.load(filino)

sig = Sigmoid().cuda()

all_preds = []
all_labels = []

for mention, context, labels in tqdm(dataloader):
# for mention, labels in tqdm(dataloader):
    mention = mention.cuda()
    context = context.cuda()
    pred = sig(model(mention, context))
    # pred = sig(model(mention))
    pred = pred.detach().cpu()

    mask = pred > .5
    batch_preds = []
    for i, m in enumerate(mask):
        ex_preds = []   
        pred_ids =  m.nonzero()

        if len(pred_ids) == 0:
            pred_ids = [torch.argmax(pred[i])]
            
        for p in pred_ids:
            ex_preds.append(id2label[p.item()])
        batch_preds.append(ex_preds)
    all_preds.extend(batch_preds)

    mask = labels == 1
    batch_labels = []
    for m in mask:
        ex_labels = []
        labels_ids = m.nonzero()
        for l in labels_ids:
            ex_labels.append(id2label[l.item()])
        batch_labels.append(ex_labels)
    all_labels.extend(batch_labels)

correct_count = defaultdict(int)
actual_count = defaultdict(int)
predict_count = defaultdict(int)

for labels, preds in zip(all_labels, all_preds):
    for pred in preds:
        predict_count[pred] += 1

        if pred in labels:
            correct_count[pred] += 1
    
    for label in labels:
        actual_count[label] += 1

def compute_f1(p, r):
    return (2*p*r)/(p + r) if p + r else 0

precisions = {k: correct_count[k]/predict_count[k] if predict_count[k] else 0 for k in label2id.keys()}
recalls = {k: correct_count[k]/actual_count[k] if actual_count[k] else 0 for k in label2id.keys()}
f1s = {k: compute_f1(precisions[k], recalls[k]) for k in label2id.keys()}

ordered_labels = list(sorted(label2id.keys()))

with open(train_dataset_stats_path, 'rb') as inp:
    train_stats = pickle.load(inp)

with open(dev_dataset_stats_path, 'rb') as inp:
    dev_stats = pickle.load(inp)

with open(metrics_file, 'a') as out:
    out.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format('label', 'train_percent', 'val_percent', 'precision', 'recall', 'f1'))
    for label in ordered_labels:
        out_string = '{}\t{:.4f}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(label,
                                                    train_stats[label],
                                                    dev_stats[label] if label in dev_stats else 0,
                                                    precisions[label],
                                                    recalls[label],
                                                    f1s[label])
        out.write(out_string)