# Code amended from evaluation_framework.py in https://github.com/mariushes/thesis_sustainability_fairness

import torch


def get_device(no_cuda=False):
    # If there's a GPU available...
    if torch.cuda.is_available() and not no_cuda:
        DEVICE_NUM = 0
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda:" + str(DEVICE_NUM))
        torch.cuda.set_device(DEVICE_NUM)
        print("Current device: ", torch.cuda.current_device())

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(DEVICE_NUM))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

# Returns a list with lists inside. Each list represents a sentence pair: [[s1,s2], [s1,s2], ...]
def get_dataset_bias_sts(data_path):
    file1 = open(data_path, 'r', encoding="utf-8")
    lines = file1.readlines()
    sentence_pairs = []
    for line in lines:
        entries = line.split("\t")
        if len(entries) > 1:  # ignore empty lines
            pair = [entries[0].replace('\n', ''), entries[1].replace('\n', ''), entries[2].replace('\n', ''), entries[3].replace('\n', '')]
            sentence_pairs.append(pair)
    return sentence_pairs


# model runs on STS-B task and returns similarity value of the sentence pair
def predict_bias_sts(sentence, sentence2, model, head_mask, tokenizer, device):
    inputs = tokenizer(sentence, sentence2, add_special_tokens=True, padding="max_length", max_length=512,
                       return_tensors='pt')
    inputs.to(device)

    # predict output tensor
    outputs = model(**inputs, head_mask=head_mask)

    # print(type(outputs))
    # print(outputs)

    return outputs[0].tolist()[0][0]
