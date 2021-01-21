from typing_model.models.BERT_models import BaseBERTTyper, OnlyContextBERTTyper, OnlyMentionBERTTyper, ConcatenatedContextBERTTyper
from torch.utils.data import DataLoader
import pickle
from torch.nn import Sigmoid 
from collections import defaultdict
import torch
from tqdm import tqdm
import numpy as np
from typing_model.losses.hierarchical_losses import HierarchicalLoss


model_path = 'checkpoints/TL_into_BBN/BF_BalancedOntonotes/model_2.ckpt'
dataloader_path = 'dataloaders/Bert_Baseline_bbn_dev.pkl'

auxiliary_variables_path =  'dataloaders/Bert_Baseline_bbn_train_auxiliary_variables.pkl'
weights_path =  'datasets_stats/ontonotes_train_weights.pkl'

metrics_file = 'result_logs/quality_prediction/TL_into_BBN/sota_BF_BalancedOntonotes.txt'

hierarchy_metrics = False
admit_void_prediction = False

with open(auxiliary_variables_path, 'rb') as filino:
  id2label, label2id, vocab_len = pickle.load(filino)

if hierarchy_metrics:
    hierarchy_path = 'experiments/ontonotes_dependency_file.tsv'
    l = HierarchicalLoss('absolute', id2label, label2id, hierarchy_path)

    no_father_value = l.NO_FATHER_VALUE
    father_dict = l.father_dict
    fathers = list(father_dict.values())

    sons_dict = defaultdict(list)

    violation_delta = 0.01

def check_violation(preds, mode):
    violation_list = []

    for p in tqdm(preds):
        violation_value = 0
        if mode == 'weak_example':
            violation_counter = 0
            ancestor_counter = 0
        

        for i in range(len(p)):
            if mode == 'weak_pred':
                violation_counter = 0
                ancestor_counter = 0

            pred_value = p[i].item()
            ancestor = father_dict[i]
            while ancestor != no_father_value and violation_value == 0:  #violation_value usefull to stop 'strong' iterations
                if mode == 'weak_example' or mode == 'weak_pred':
                    ancestor_counter += 1
                ancestor_value = p[ancestor].item()
                if pred_value > ancestor_value + violation_delta:
                    if mode == 'strong':
                        violation_value = 1
                    elif mode == 'weak_example' or mode == 'weak_pred':
                        violation_counter += 1
                
                ancestor = father_dict[ancestor]
                # BREAKPOINT
            if mode == 'weak_pred' and ancestor_counter > 0:
                violation_list.append(violation_counter/ancestor_counter)

        if mode == 'strong':
            violation_list.append(violation_value)
        elif mode == 'weak_example':
            violation_list.append(violation_counter/ancestor_counter)
        # BREAKPOINT
    return violation_list

with open(weights_path, 'rb') as inp:
    # weights = pickle.load(inp)
    # ordered_weights = torch.tensor([weights[id2label[i]] for i in range(len(id2label))])    
    ordered_weights = None    
    model = ConcatenatedContextBERTTyper.load_from_checkpoint(model_path, classes = vocab_len, id2label = id2label, label2id = label2id, weights = ordered_weights)
    model.cuda()
    model.eval()

with open(dataloader_path, "rb") as filino:
    dataloader = pickle.load(filino)

sig = Sigmoid().cuda()

all_preds = []
all_labels = []

void_counter = 0
avg_predictions = 0

strong_violation_list = []
weak_example_violation_list = [] 
weak_pred_violation_list = [] 

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
            if admit_void_prediction:
                void_counter += 1
            else:
                pred_ids = [torch.argmax(pred[i])]
        avg_predictions += len(pred_ids)

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

    #compute hierarchy violation
    if hierarchy_metrics:
        strong_violation_list.extend(check_violation(pred, 'strong'))
        weak_example_violation_list.extend(check_violation(pred, 'weak_example'))
        weak_pred_violation_list.extend(check_violation(pred, 'weak_pred'))

if hierarchy_metrics:
    strong_violation_value = np.mean(strong_violation_list)
    weak_example_violation_value = np.mean(weak_example_violation_list)
    weak_pred_violation_value = np.mean(weak_pred_violation_list)

avg_predictions = avg_predictions / len(all_labels)

correct_count = defaultdict(int)
actual_count = defaultdict(int)
predict_count = defaultdict(int)

pred_counter = 0
label_counter = 0
correct_counter = 0

precisions = []
recalls = []

for labels, preds in zip(all_labels, all_preds):

    correct_labels = len(set(labels).intersection(set(preds)))
    correct_counter += correct_labels
    pred_counter += len(preds)
    label_counter += len(labels)

    precisions.append(correct_labels/len(preds) if len(preds) else 0)
    recalls.append(correct_labels/len(labels))

    for pred in preds:
        predict_count[pred] += 1

        if pred in labels:
            correct_count[pred] += 1
    
    for label in labels:
        actual_count[label] += 1

def compute_f1(p, r):
    return (2*p*r)/(p + r) if p + r else 0

macro_p_examples = np.mean(precisions)
macro_r_examples = np.mean(recalls)
macro_f1_examples = compute_f1(macro_p_examples, macro_r_examples)

micro_p = correct_counter/pred_counter
micro_r = correct_counter/label_counter
micro_f1 = compute_f1(micro_p, micro_r)

precisions = {k: correct_count[k]/predict_count[k] if predict_count[k] else 0 for k in label2id.keys()}
recalls = {k: correct_count[k]/actual_count[k] if actual_count[k] else 0 for k in label2id.keys()}

macro_p_classes = np.mean(list(precisions.values()))
macro_r_classes = np.mean(list(recalls.values()))
macro_f1_classes = compute_f1(macro_p_classes, macro_r_classes)


with open(metrics_file, 'a') as out:
    if hierarchy_metrics:
        out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(micro_f1, macro_f1_examples, macro_f1_classes,  
                                                            avg_predictions, void_counter,
                                                            strong_violation_value, weak_example_violation_value, weak_pred_violation_value))
    else:
        out.write('{}\t{}\t{}\t{}\t{}\n'.format(micro_f1, macro_f1_examples, macro_f1_classes,  
                                                            avg_predictions, void_counter))
   