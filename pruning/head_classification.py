import sys
import classify_normed_patterns
sys.path.insert(0, '../src')

#import classify_attention_patterns
#import classify_normed_patterns
from argparse import Namespace
from scratch import load_and_cache_examples, set_seed
from transformers import RobertaConfig, RobertaTokenizer
from transformers import RobertaForSequenceClassification, RobertaForMaskedLM
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
import torch
import random
from collections import Counter
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

head_classifier_model, label2id, min_max_size = classify_attention_patterns.load_model("../models/head_classifier/classify_attention_patters.tar")
head_classifier_model = head_classifier_model.eval().cuda()
id2label = {idx:label for label, idx in label2id.items()}

normed_head_classifier_model, normed_label2id, normed_min_max_size = classify_normed_patterns.load_model(
 "../models/head_classifier/classify_normed_patterns.tar")
normed_head_classifier_model = normed_head_classifier_model.eval().cuda()
normed_id2label = {idx: label for label, idx in normed_label2id.items()}

### FineTune Model
set_seed(1337)
for task in ["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI", "QNLI", "RTE"]:
    attention_counter = Counter()
    normed_attention_counter = Counter()
    for seed in ["seed_1337", "seed_42", "seed_86", "seed_71", "seed_166"]:
        # Load Model
        model_path = f"../models/finetuned/{task}/{seed}/"
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        config = RobertaConfig.from_pretrained(model_path)
        config.output_attentions = True
        transformer_model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)
        transformer_model = transformer_model.eval()
        transformer_model.cuda()
        args = Namespace(data_dir=f"../data/glue/{task}/", local_rank=-1,
                         model_name_or_path=model_path,
                         overwrite_cache=False, model_type="bert", max_seq_length=128)
        eval_dataset = load_and_cache_examples(args, task.lower(), tokenizer, evaluate=True)
        eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)
        input_data = None
        attention_head_types = [[] for _ in range(12)]
        normed_attention_head_types = [[] for _ in range(12)]

        # Prune
        mask_path = f"../masks/heads_mlps_super/{task}/{seed}/"
        head_mask = np.load(f"{mask_path}/head_mask.npy")
        mlp_mask = np.load(f"{mask_path}/mlp_mask.npy")
        head_mask = torch.from_numpy(head_mask)
        heads_to_prune = {}
        for layer in range(len(head_mask)):
            heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
            heads_to_prune[layer] = heads_to_mask
        mlps_to_prune = [h[0] for h in (1 - torch.from_numpy(mlp_mask).long()).nonzero().tolist()]

        transformer_model.prune_heads(heads_to_prune)
        transformer_model.prune_mlps(mlps_to_prune)
        transformer_model = transformer_model.eval()
        transformer_model.cuda()
        args = Namespace(data_dir=f"../data/glue/{task}/", local_rank=-1,
                         model_name_or_path=model_path,
                         overwrite_cache=False, model_type="bert", max_seq_length=128)
        eval_dataset = load_and_cache_examples(args, task.lower(), tokenizer, evaluate=True)
        eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)
        input_data = None
        attention_head_types = [[] for _ in range(12)]
        normed_attention_head_types = [[] for _ in range(12)]

        k = 0
        for batch in eval_dataloader:
            batch = tuple(t.to("cuda:0") for t in batch)
            n_tokens = batch[1].sum()
            if n_tokens < min_max_size[0]:
                continue
            input_data = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            with torch.no_grad():
                _, _, attentions, allalpha_f, attention_mask = transformer_model(**input_data)
                for layer in range(len(attentions)):
                    if attentions[layer] is None:
                        continue

                    num_classifier_tokens = max(n_tokens, min_max_size[0] + 20)
                    head_attentions = attentions[layer].transpose(0, 1)

                    head_attentions = head_attentions[:, :, :num_classifier_tokens, :num_classifier_tokens]
                    logits = head_classifier_model(head_attentions)
                    label_ids = torch.argmax(logits, dim=-1)
                    labels = [id2label[int(label_id.item())] for label_id in label_ids]
                    if len(attention_head_types[layer]) == 0:
                        for i in range(len(labels)):
                            attention_head_types[layer].append(Counter())
                    for i, label in enumerate(labels):
                        attention_head_types[layer][i][label] += 1

                    num_classifier_tokens = max(n_tokens, normed_min_max_size[0] + 20)
                    normed_alpha_f = allalpha_f[layer].norm(dim=-1)
                    normed_head_attentions = normed_alpha_f.transpose(0, 1)

                    normed_head_attentions = normed_head_attentions[:, :, :num_classifier_tokens,
                                             :num_classifier_tokens]
                    logits = normed_head_classifier_model(normed_head_attentions)
                    label_ids = torch.argmax(logits, dim=-1)
                    labels = [normed_id2label[int(label_id.item())] for label_id in label_ids]
                    if len(normed_attention_head_types[layer]) == 0:
                        for i in range(len(labels)):
                            normed_attention_head_types[layer].append(Counter())
                    for i, label in enumerate(labels):
                        normed_attention_head_types[layer][i][label] += 1
                del attentions, allalpha_f
            k += 1
            if k == 100:
                break
        for layer in attention_head_types:
            for head_type_ctr in layer:
                attention_counter += head_type_ctr
        total_counter = Counter()
        for layer in normed_attention_head_types:
            for head_type_ctr in layer:
                normed_attention_counter += head_type_ctr
    print(task)
    print("attention:", {k: v / sum(attention_counter.values()) for k, v in attention_counter.most_common()})
    print("weight normed:", {k:v/sum(normed_attention_counter.values()) for k,v in normed_attention_counter.most_common()})