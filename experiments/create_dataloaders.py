from typing_model.runner.base_experimenters import BaseTypingExperimentClass
from typing_model.data_models.base_dataclass import BaseDataclass
from typing_model.data.BERT_datasets import ConcatenatedContextTypingBERTDataSet
import configparser
import pickle

config_file = 'experiments/create_dataloaders.ini'

config = configparser.ConfigParser()
config.read(config_file)

dataclass = BaseDataclass(**config['FIGER'])

b = BaseTypingExperimentClass(dataclass)

with open(b.auxiliary_variables_path, 'rb') as filino:
	id2label, label2id, vocab_len = pickle.load(filino)

path_dict = {
			 'test': b.test_data_path}
b.dataset_class = ConcatenatedContextTypingBERTDataSet
b.get_dataloader_from_dataset_path(path_dict, 
                                    'test', 
                                    load_path=False,
                                    load_variables=True, 
                                    save_path=b.save_test_dataset_path,
                                    id2label=id2label,
                                    label2id=label2id,
                                    vocab_len=vocab_len)