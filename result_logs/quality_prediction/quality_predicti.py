# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
path = 'TL_into_Ontonotes/sota_BF_FIGER.txt'
out_path = 'averages.txt'


# %%
with open(path, 'r') as inp:
    lines = [l.replace('\n', '').split('\t') for l in inp.readlines()]
    lines = [[float(v) for v in l]for l in lines]


# %%
import numpy as np
metric_number = 5

avg_metrics = ''
stds = ''

for i in range(metric_number):
    avg_metrics += '{}\t'.format(np.mean([l[i] for l in lines]))
    stds += '{}\t'.format(np.std([l[i] for l in lines]))

with open(out_path, 'a') as out:
    out.write('\n--------------------------------\n')
    out.write(avg_metrics[:-1])
    out.write('\n')
    out.write(stds[:-1])


# %%
np.mean([69, 69, 115])


# %%
import numpy as np

np.std([69, 69, 115])


# %%



