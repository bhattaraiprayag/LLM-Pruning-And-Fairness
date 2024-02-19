from datasets import load_dataset, DatasetDict

# needed for testing
from transformers import glue_processors as processors
from transformers import InputExample
import csv


def save_ds(datadict, task, model_no):
    dir = f'{task}/model_no{model_no}'
    datadict['train'].to_csv(f'{dir}/train.tsv', sep='\t')
    datadict['validation'].to_csv(f'{dir}/dev.tsv', sep='\t')
    datadict['test'].to_csv(f'{dir}/test.tsv', sep='\t')

    if task == 'mnli':
        datadict['validation_mismatched'].to_csv(f'{dir}/dev_mm.tsv', sep='\t')
        datadict['test_mismatched'].to_csv(f'{dir}/test_mm.tsv', sep='\t')


# Step 1: create a DatasetDict for every model using the right splits

# STS-B

# model_no1
stsb_1 = load_dataset(
    "glue",
    "stsb",
    split=['train', 'validation[:50%]', 'validation[-50%:]']
)
stsb_1 = DatasetDict({'train': stsb_1[0], 'validation': stsb_1[1], 'test': stsb_1[2]})

# model_no2
stsb_2 = load_dataset(
    "glue",
    "stsb",
    split='train+validation'
)
# 20% test + validation (keep the same ratio as in the original split)
train_testvalid = stsb_2.train_test_split(test_size=0.2, shuffle=True, seed=2)
# Split test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=True, seed=2)
# gather everything into a single DatasetDict
stsb_2 = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']})

# model_no3
stsb_3 = load_dataset(
    "glue",
    "stsb",
    split='train+validation'
)
# 20% test + validation (keep the same ratio as in the original split)
train_testvalid = stsb_3.train_test_split(test_size=0.2, shuffle=True, seed=3)
# Split test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=True, seed=3)
# gather everything into a single DatasetDict
stsb_3 = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']})

# MNLI

# model_no1
mnli_1 = load_dataset(
    "glue",
    "mnli",
    split=['train', 'validation_matched[:50%]', 'validation_mismatched[:50%]', 'validation_matched[-50%:]',
           'validation_mismatched[-50%:]']
)

mnli_1 = DatasetDict({
    'train': mnli_1[0],
    'validation': mnli_1[1],
    'validation_mismatched': mnli_1[2],
    'test': mnli_1[3],
    'test_mismatched': mnli_1[4]
})

# model_no2
mnli_2 = load_dataset(
    "glue",
    "mnli",
    split=['train+validation_matched', 'validation_mismatched[:50%]', 'validation_mismatched[-50%:]']
)
# 2.5% test + validation (keep the same ratio as in the original split
train_testvalid = mnli_2[0].train_test_split(test_size=0.025, shuffle=True, seed=2)
# Split the test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=True, seed=2)
# turn datasets into a DatasetDict
mnli_2 = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train'],
    'test_mismatched': mnli_2[1],
    'validation_mismatched': mnli_2[2]
})

# model_no3
mnli_3 = load_dataset(
    "glue",
    "mnli",
    split=['train+validation_matched', 'validation_mismatched[:50%]', 'validation_mismatched[-50%:]']
)
# 2.5% test + validation (keep the same ratio as in the original split
train_testvalid = mnli_3[0].train_test_split(test_size=0.025, shuffle=True, seed=3)
# Split the test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=True, seed=3)
# turn datasets into a DatasetDict
mnli_3 = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train'],
    'test_mismatched': mnli_3[1],
    'validation_mismatched': mnli_3[2]
})

# Step 2: Save all splits locally in tsv files

# STS-B
save_ds(stsb_1, 'sts-b', 1)
save_ds(stsb_2, 'sts-b', 2)
save_ds(stsb_3, 'sts-b', 3)

# MNLI
save_ds(mnli_1, 'mnli', 1)
save_ds(mnli_2, 'mnli', 2)
save_ds(mnli_3, 'mnli', 3)

# testing
'''def create_examples(lines):
    """Creates examples for dev set."""
    examples = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        guid = f"dev-{line[-1]}"
        text_a = line[0]
        text_b = line[1]
        label = line[2]
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

with open('sts-b/model_no1/dev.tsv', "r", encoding="utf-8-sig") as f:
    data = list(csv.reader(f, delimiter="\t"))
    examples = (
        create_examples(data)
    )
print(examples)'''
