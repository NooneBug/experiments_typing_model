from collections import defaultdict
import numpy
import pickle
from typing_model.losses.hierarchical_losses import HierarchicalLoss

auxiliary_variables_path =  '../typing_experiments/dataloaders/simpler_ontonotes_train_auxiliary_variables.pkl'

hierarchy_path = 'experiments/ontonotes_dependency_file.tsv'
with open(auxiliary_variables_path, 'rb') as filino:
  id2label, label2id, vocab_len = pickle.load(filino)


l = HierarchicalLoss('absolute', id2label, label2id, hierarchy_path)
no_father_value = l.NO_FATHER_VALUE
father_dict = l.father_dict
fathers = list(father_dict.values())

sons_dict = defaultdict(list)

for k, v in father_dict.items():
  sons_dict[v].append(k)

def down_leaf_adjustment(predictions):
  adj_preds = []
  for pred in predictions:
    adj = []
    for elem in range(len(predictions)):
      father = father_dict[elem]
      if elem not in fathers and father != no_father_value: 
        if pred[elem] > pred[father]:
          to_ins = pred[father].item()
        else:
          to_ins = pred[elem].item()
      else:            
        to_ins = pred[elem].item()
      adj.append(to_ins)
    adj_preds.append(numpy.array(adj))
  return adj_preds 

def up_leaf_adjustment(predictions):
  adj_preds = []
  for pred in predictions:
    adj = []
    for elem in range(len(predictions)):
      father = father_dict[elem]
      max_sibling_logit = 0

      for i in sons_dict[father]:
        adj.append(pred[i])

        if i not in fathers and father != no_father_value:
          if pred[i] > pred[father] and pred[i] > max_sibling_logit:
            max_sibling_logit = pred[i]
      
      if max_sibling_logit > 0:
        adj[father] = max_sibling_logit

    adj_preds.append(numpy.array(adj))
  return adj_preds 

def top_down_child_adjustment(predictions):
  adj_preds = []
  for pred in predictions:
    adj = []
    for elem in range(len(predictions)):
      father = father_dict[elem]  
      adj.append(pred[elem])
      if father != no_father_value: 
        if adj[elem] > adj[father]:
          to_ins = adj[father]
        else:
          to_ins = adj[elem]
      else:            
          to_ins = adj[elem]
      adj[elem] = to_ins
    adj_preds.append(numpy.array(adj))
  return adj_preds 
  
# def bottom_up_father_adjustment(predictions):
#   adj_preds = []
#   for pred in predictions:
#     for j, _ in enumerate(hierarchy_indexes):
#       elem = hierarchy_indexes[-j-1] #pick elements from the last one
#       start, end, father = elem[0], elem[1], elem[2]
#       max_sibling_logit = 0

#       for i in range(start, end + 1):
#         if father != 'null':
#           if pred[i] > pred[father] and pred[i] > max_sibling_logit:
#             max_sibling_logit = pred[i]
      
#       if max_sibling_logit > 0:
#         pred[father] = max_sibling_logit

#     adj_preds.append(numpy.array(pred))
#   return adj_preds 
  
# def conditional_probability_weighting(predictions):
#   adj_preds = []
#   for pred in predictions:
#     for elem in hierarchy_indexes:
#       start, end, father = elem[0], elem[1], elem[2]
#       for i in range(start, end + 1):
#         if father != 'null': 
#           pred[i] = pred[i] * pred[father]
#     adj_preds.append(numpy.array(pred))
#   return adj_preds 
