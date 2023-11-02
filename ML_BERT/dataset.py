from torch.utils.data import Dataset
import pandas as pd
import torch


class BertDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        super(BertDataset, self).__init__()

        # store the bert tokenizer
        self.tokenizer = tokenizer

        # read the tsv file
        data = pd.read_csv(data_path, delimiter='\t', header=None, encoding='utf-16')

        # remove nans in the dataset
        data.dropna(axis=0, inplace=True)
        data = data.iloc[0:1000]

        # store the text data
        self.text_data = data.iloc[:, 1].tolist()

        count = 0
        for text in self.text_data:

            if type(text) != str:
                count += 1

        # extract the labels
        self.targets = self.convert_labels(data.iloc[:, 0].tolist())

        # max length of the text sequence
        self.max_length = max_length

    @staticmethod
    def convert_labels(label_list: list) -> list:

        label_list = [0 if x == 'de' else x for x in label_list]
        label_list = [1 if x == 'bar' else x for x in label_list]
        label_list = [2 if x == 'nds' else x for x in label_list]
        return label_list

    def __len__(self) -> int:

        # function which returns the length of the dataset
        return len(self.text_data)

    def __getitem__(self, index: int) -> dict:

        # extract the text sample at the specified index
        text = self.text_data[index]
        try:
            # tokenize the text sample
            inputs = self.tokenizer.encode_plus(
                text,
                padding='max_length',
                add_special_tokens=True,
                return_attention_mask=True,
                truncation=True
            )

            # token ids
            ids = inputs["input_ids"]

            # token type ids
            token_type_ids = inputs["token_type_ids"]

            # attention mask
            mask = inputs["attention_mask"]

            # dictionary in which the output is stored
            data_dict = {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'target': torch.tensor(self.targets[index], dtype=torch.long)
            }

            return data_dict
        except:
            print(text)
