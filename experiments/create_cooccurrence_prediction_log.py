from typing_model.models.BERT_models import BaseBERTTyper, TransformerBERTTyper, ConcatenatedContextBERTTyper, OnlyContextBERTTyper, OnlyMentionBERTTyper
from torch.utils.data import DataLoader
import pickle
from torch.nn import Sigmoid 
from collections import defaultdict
import torch
from tqdm import tqdm
# from experiments.adjustments import down_leaf_adjustment, up_leaf_adjustment, top_down_child_adjustment


model_path = '../typing_experiments/checkpoints/only_context-v0.ckpt'
dataloader_path = '../typing_experiments/dataloaders/only_context_ontonotes_dev.pkl'
auxiliary_variables_path =  '../typing_experiments/dataloaders/only_context_ontonotes_train_auxiliary_variables.pkl'
weights_path =  '../typing_experiments/datasets_stats/ontonotes_train_weights.pkl'

train_dataset_stats_path = '../typing_experiments/datasets_stats/ontonotes_train.pkl'
dev_dataset_stats_path = '../typing_experiments/datasets_stats/ontonotes_dev.pkl'

model_name = 'OnlyContext'

n = 5

with open(auxiliary_variables_path, 'rb') as filino:
    id2label, label2id, vocab_len = pickle.load(filino)
with open(weights_path, 'rb') as inp:
    weights = pickle.load(inp)
    # ordered_weights = torch.tensor([weights[id2label[i]] for i in range(len(id2label))])    
    ordered_weights = None   
    model = OnlyContextBERTTyper.load_from_checkpoint(model_path, classes = vocab_len, id2label = id2label, label2id = label2id, weights = ordered_weights)
    model.cuda()
    model.eval()
# model = BaseBERTTyper.load_from_checkpoint(model_path, classes = vocab_len, id2label = id2label, label2id = label2id)

with open(dataloader_path, "rb") as filino:
    dataloader = pickle.load(filino)

sig = Sigmoid().cuda()


dla = False
ula = False
tdca = False
bufa = False
cpw = False

print('-----------------------------------------')
print('dla: {}, ula: {}, tdca: {}, bufa: {}, cpw: {}'.format(dla, ula, tdca, bufa, cpw))
print('-----------------------------------------')


adj_name = ''

adjs = [dla, ula, tdca, bufa, cpw]

for a in adjs:
  for b in adjs:
    if a != b and a and b:
      raise Exception('pls select only one adjustment method between dla, ula, tdca, bufa, cpw')

all_labels = []
all_preds = []
all_logits = []

# for mention, context, labels in tqdm(dataloader):
for mention, labels in tqdm(dataloader):
    mention = mention.cuda()
    # context = context.cuda()
    # pred = sig(model(mention, context))
    pred = sig(model(mention))
    pred = pred.detach().cpu()

    mask = pred > .5
    batch_preds = []
    for m in mask:
        ex_preds = []   
        pred_ids =  m.nonzero()
        for p in pred_ids:
            ex_preds.append(id2label[p.item()])
        batch_preds.append(ex_preds)
    all_preds.extend(batch_preds)

    all_logits.extend(pred)

    mask = labels == 1
    batch_labels = []
    for m in mask:
        ex_labels = []
        labels_ids = m.nonzero()
        for l in labels_ids:
            ex_labels.append(id2label[l.item()])
        batch_labels.append(ex_labels)
    all_labels.extend(batch_labels)

adj_preds = []
if dla:
  adj_name = '_dla'
  adj_preds = down_leaf_adjustment(all_logits)

if ula:
  adj_name = '_ula'
  adj_preds = up_leaf_adjustment(all_logits)

if tdca:
  adj_name = '_tdca'
  adj_preds = top_down_child_adjustment(all_logits)

# if bufa:
#   adj_name = '_bufa'
#   adj_preds = bottom_up_father_adjustment(all_logits)

# if cpw:
#   adj_name = '_cpw'
#   adj_preds = conditional_probability_weighting(all_logits)

if adj_preds:
  print('generate the adjusted predictions: ...')
  all_preds = []

  for single_pred in adj_preds:
    mask = single_pred > 0.5
    labels_pred = [id2label[i] for i, v in enumerate(mask) if v]
    all_preds.append(labels_pred)


predicted_types = {label: {} for label in id2label.values()}
correct_types = {label: {} for label in id2label.values()}
wrong_types = {label: {} for label in id2label.values()}

absolute_prediction_counter = {label: 0 for label in id2label.values()}
correct_prediction_counter = {label: 0 for label in id2label.values()}
wrong_prediction_counter = {label: 0 for label in id2label.values()}

precisions = {}
recalls = {}
f1s = {}

true_type_counter = {label: 0 for label in id2label.values()}

bar = tqdm(total = len(all_preds))
bar.set_description('compute_metrics')


for pred, true in zip(all_preds, all_labels):
  for true_label in true:
    true_type_counter[true_label] += 1
    for pred_label in pred:
      # update predicted_types
      try:
        predicted_types[true_label][pred_label] += 1
      except:
        predicted_types[true_label][pred_label] = 1
      
      if pred_label in true:
        #update correct_types
        try:
          correct_types[true_label][pred_label] += 1
        except:
          correct_types[true_label][pred_label] = 1
      else:
        #update wrong_types
        try:
          wrong_types[true_label][pred_label] += 1
        except:
          wrong_types[true_label][pred_label] = 1
  
  # compute totals for precision and recall
  for pred_label in pred:
    absolute_prediction_counter[pred_label] += 1
    if pred_label in true:
      correct_prediction_counter[pred_label] += 1

for label, count in true_type_counter.items():
  if count:
    if absolute_prediction_counter[label]:
      precisions[label] = correct_prediction_counter[label]/absolute_prediction_counter[label]
    else:
      precisions[label] = 0
    recalls[label] = correct_prediction_counter[label]/true_type_counter[label]
    if precisions[label] != 0 and recalls[label] != 0:
      f1s[label] = 2/((1/precisions[label]) + (1/recalls[label]))
    else:
      f1s[label] = 0
                
# compute relative values and sort
normalized_predicted_types = {label: dict(sorted({pred_label: count/true_type_counter[label] for pred_label, count in pred_counter.items() if true_type_counter[label]}.items(), key=lambda x:x[1], reverse = True)) for label, pred_counter in predicted_types.items()}
absolute_correct_types = {label: dict(sorted({pred_label: count/true_type_counter[label] for pred_label, count in correct_counter.items()}.items(),key = lambda x: x[1], reverse = True)) for label, correct_counter in correct_types.items()}
relative_correct_types = {label: {pred_label: count/predicted_types[label][pred_label] for pred_label, count in correct_counter.items()} for label, correct_counter in correct_types.items()}
absolute_wrong_types = {label: dict(sorted({pred_label: count/true_type_counter[label] for pred_label, count in wrong_counter.items()}.items(),key = lambda x: x[1], reverse = True)) for label, wrong_counter in wrong_types.items()}
relative_wrong_types = {label: {pred_label: count/predicted_types[label][pred_label] for pred_label, count in wrong_counter.items()} for label, wrong_counter in wrong_types.items()}

def load_data_with_pickle(path):
  with open(path, 'rb') as file:
    return pickle.load(file)

ordered_type_cooc = load_data_with_pickle('datasets_stats/ontonotes_cooccurrence_stats.pkl')
ordered_dev_type_cooc = load_data_with_pickle('datasets_stats/ontonotes_dev_cooccurrence_stats.pkl')

true_type_counter = {k : v for k, v in true_type_counter.items() if v != 0}
sorted_true_type_counter = dict(sorted(true_type_counter.items(), key=lambda x:x[1], reverse=True))


with open('../typing_experiments/result_logs/quality_prediction/cooc_{}{}_prediction_log.txt'.format(model_name, adj_name), 'a') as out:
  out.write('\nocc\tfocused_type\tp\tr\tf1\tmctt_type\tmctt_%\tmctv_type\tmctv_%\tmctp_type\tmctp_%mcctp_type\tmcctp_abs\tmcctp_rel\tmcwtp_abs\tmcwtp_rel')
  for label, occurrences in sorted_true_type_counter.items():
    try:
      for i in range(n):
        if i == 0:
          out.write('\n{}\t{}\t{}\t{}\t{}'.format(occurrences, 
                              label + '_{}'.format(i + 1), 
                              precisions[label], recalls[label], f1s[label])
                            )
        else:
          out.write('\n\t{}\t\t\t'.format(label + '_{}'.format(i + 1)))
          
        if label in ordered_type_cooc:  
          if i < len(ordered_type_cooc[label]):
            out.write('\t{}\t{}'.format(list(ordered_type_cooc[label].keys())[i],
                            list(ordered_type_cooc[label].values())[i],
                            ))
          else:
            out.write('\t\t')
        else:
          out.write('\t\t')
        
        if i < len(ordered_dev_type_cooc[label]):
          out.write('\t{}\t{}'.format(list(ordered_dev_type_cooc[label].keys())[i],
                          list(ordered_dev_type_cooc[label].values())[i],
                          ))
        else:
          out.write('\t\t')
        
        
        if i < len(normalized_predicted_types[label]):
          out.write('\t{}\t{}'.format(list(normalized_predicted_types[label].keys())[i],
                        list(normalized_predicted_types[label].values())[i]
                        ))
        else:
          out.write('\t\t')
        if i < len(absolute_correct_types[label]):
          absolute_correct_i_label = list(absolute_correct_types[label].keys())[i]
          out.write('\t{}\t{}\t{}'.format(absolute_correct_i_label,
                          absolute_correct_types[label][absolute_correct_i_label],
                          relative_correct_types[label][absolute_correct_i_label]
                            ))
        else:
          out.write('\t\t\t')
        if i < len(absolute_wrong_types[label]):
          absolute_wrong_i_label = list(absolute_wrong_types[label].keys())[i]
          out.write('\t{}\t{}\t{}'.format(absolute_wrong_i_label,
                          absolute_wrong_types[label][absolute_wrong_i_label],
                          relative_wrong_types[label][absolute_wrong_i_label],
                            ))   
        else:
          out.write('\t\t\t')
    except:
      pass
