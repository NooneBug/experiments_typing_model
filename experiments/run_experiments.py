from typing_model.runner.base_experimenters import ExperimentRoutine
from typing_model.runner.BERT_experimenters import ConcatenatedContextBERTTyperExperiment,\
    BertHierarchicalExperiment, BertHierarchicalRegularizedExperiment, BertOnlyMentionExperiment, \
        BertOnlyContextExperiment
from typing_model.runner.Elmo_experimenters import ElmoBaseExperiment

from typing_model.data_models.base_dataclass import BaseDataclass, ElmoDataclass


# config_file_path = './experiments/configs/B_ontonotes.ini'
# config_file_path = './experiments/configs/B_choi.ini'
# config_file_path = './experiments/configs/B_FIGER.ini'
# config_file_path = './experiments/configs/B_balanced_ontonotes.ini'
config_file_path = './experiments/configs/FT_into_FIGER.ini'
# config_file_path = './experiments/configs/FT_into_Balanced_Ontonotes.ini'
# config_file_path = './experiments/configs/FT_into_Ontonotes.ini'
# config_file_path = './experiments/configs/FT_into_BBN.ini'

# exp_list is a list of experiment configurations: each element has to be a dict with:
# exp_name: a tag present in the config file at `config_file_path`
# Dataclass: a dataclass which can take in input the parameters in the above configfile at the `exp_name` tag
# ExperimentClass: a class which follows the typing_model.runner.experimenters.BaseExperimentClass interface
exp_list = [
            {
                'exp_name': name,
                'Dataclass': BaseDataclass,
                'ExperimentClass': ConcatenatedContextBERTTyperExperiment
            } for name in ['B_OntonotesIntoFIGER0', 'BF_OntonotesIntoFIGER0',
                            'B_OntonotesIntoFIGER1', 'BF_OntonotesIntoFIGER1',
                            'B_OntonotesIntoFIGER2', 'BF_OntonotesIntoFIGER2']     
        ]

exp_routine = ExperimentRoutine(exp_list = exp_list, config_file=config_file_path)
exp_routine.perform_experiments()