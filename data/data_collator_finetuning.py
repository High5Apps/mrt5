# data_collator_finetuning.py
# Author: Julie Kallini

class XNLIDataCollator:
    def __init__(self, tokenizer, padding='longest', max_length=1024, truncation=True):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation
        # 0: entailment, 1: contradiction, 2: neutral
        self.label_map = {0: '0', 1: '1', 2: '2'}

    def __call__(self, features=None):
        input_texts = []
        label_texts = []

        for feature in features:
            input_text = f"Premise: {feature['premise']} Hypothesis: {feature['hypothesis']}"
            label_text = self.label_map[feature['label']]
            input_texts.append(input_text)
            label_texts.append(label_text)

        # Tokenize inputs and labels for the entire batch
        inputs_batch = self.tokenizer(input_texts, padding=self.padding, truncation=self.truncation,
                                      max_length=self.max_length, return_tensors='pt')
        labels_batch = self.tokenizer(label_texts, padding=self.padding, truncation=self.truncation,
                                      max_length=2, return_tensors='pt')

        batch = {}
        batch['input_ids'] = inputs_batch["input_ids"]
        batch['attention_mask'] = inputs_batch["attention_mask"]
        batch['labels'] = labels_batch["input_ids"]

        return batch
