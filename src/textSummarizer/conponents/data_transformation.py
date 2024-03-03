import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk,DatasetDict
from textSummarizer.config.configuration import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    
    def convert_examples_to_features(self,example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'] , max_length = 1024, truncation = True )
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length = 128, truncation = True )
            
        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
         # Efficiently select the first 10 rows using `select()`
        
        ### because I am using cpu ###

        # Select only the first three rows from the "train" split
        train_dataset = dataset_samsum["train"].select([0, 1])
        
        # Select only the first three rows from the "validation" split
        validation_dataset = dataset_samsum["validation"].select([0])
        
        # Create a DatasetDict with "train" and "validation" keys
        dataset_samsum = DatasetDict({
            "train": train_dataset,
            "validation": validation_dataset,
        })

        #dataset_samsum = dataset_samsum.select(range(10)) 
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir,"samsum_dataset"))