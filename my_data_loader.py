import torch
import numpy as np
import random


class MyDataLoader(object):
    def __init__(self, actions_dict, data_breakfast, labels_breakfast):
        list_of_examples = list(range(len(data_breakfast)))
        random.shuffle(list_of_examples)

        self.test_list_len = int(0.9 * len(list_of_examples))
        self.test_list = list_of_examples[:self.test_list_len]
        self.validation_list = list_of_examples[self.test_list_len:]

        self.current_index = 0
        self.num_classes = len(actions_dict)
        self.actions_dict = actions_dict

        self.data_breakfast = data_breakfast
        self.labels_breakfast = labels_breakfast

    def reset(self):
        self.current_index = 0
        random.shuffle(self.test_list)

    def has_next_test(self):
        if self.current_index < len(self.test_list):
            return True
        return False

    def next_test_batch(self, batch_size):
        batch_indexes = self.test_list[self.current_index:self.current_index + batch_size]
        self.current_index += batch_size

        batch_input = []
        batch_target = []
        for index in batch_indexes:
            batch_input.append(self.data_breakfast[index].transpose(1, 0))
            batch_target.append(self.labels_breakfast[index])

        length_of_sequences = map(len, batch_target)
        max_length_of_sequences = max(length_of_sequences)

        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max_length_of_sequences, dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max_length_of_sequences, dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max_length_of_sequences, dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = batch_input[i]
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = batch_target[i]
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
