import evaluate
import torch

from datasets import load_dataset
from evaluation.utils.bias_sts import get_device


def load_test_dataset(task, model_no):
    """Loads the test dataset based on the specified task."""
    if task == 'mnli':
        if model_no == 1:  # uses original split of the data
            return (
                load_dataset('glue', 'mnli', split='validation_matched[-50%:]'),
                load_dataset('glue', 'mnli', split='validation_mismatched[-50%:]')
            )
        else:  # uses shuffled split based on seed
            full_dataset = load_dataset(
                "glue",
                "mnli",
                split=['train+validation_matched', 'validation_mismatched[:50%]', 'validation_mismatched[-50%:]']
            )
            # 2.5% test_matched + validation_matched (keep the same ratio as in the original split)
            train_testvalid = full_dataset[0].train_test_split(test_size=0.025, shuffle=True, seed=model_no)
            # Split test_matched + validation_matched in half test_matched, half validation_matched
            test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=True, seed=model_no)
            # return test portion (matched and mismatchend)
            return (
                test_valid['train'],
                full_dataset[2]
            )

    elif task == 'stsb':
        if model_no == 1:  # uses original split of the data
            return load_dataset('glue', 'stsb', split='validation[-50%:]')
        else:  # uses shuffled split based on seed
            full_dataset = load_dataset(
                "glue",
                "stsb",
                split='train+validation'
            )
            # 20% test + validation (keep the same ratio as in the original split)
            train_testvalid = full_dataset.train_test_split(test_size=0.2, shuffle=True, seed=model_no)
            # Split test + valid in half test, half valid
            test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=True, seed=model_no)
            # return test portion
            return test_valid['test']

    else:
        raise ValueError(f'No evaluation dataset found for task {task}')


def evaluate_metrics(model, head_mask, tokenizer, task, test_dataset):
    """Evaluates task-specific metrics and returns results."""
    results_dict = {}
    if task == 'mnli':
        eval_matched, eval_mismatched = test_dataset
        mnli_matched = evaluate_model(model, head_mask, tokenizer, task, eval_matched)
        mnli_mismatched = evaluate_model(model, head_mask, tokenizer, task, eval_mismatched)
        results_dict['Matched Acc'], results_dict['Mismatched Acc'] = mnli_matched['accuracy'], mnli_mismatched[
            'accuracy']
    elif task == 'stsb':
        eval_results = evaluate_model(model, head_mask, tokenizer, task, test_dataset)
        results_dict['Spearmanr'], results_dict['Pearson'] = eval_results['spearmanr'], eval_results['pearson']
    else:
        raise ValueError(f'No evaluation metrics found for task {task}')

    return results_dict


def evaluate_model(model, head_mask, tokenizer, task_name, test_dataset):
    device = get_device()

    # define the names of the sentence keys based on task
    if task_name == "mnli":
        sent1, sent2 = "premise", "hypothesis"
    else:  # STS-B
        sent1, sent2 = "sentence1", "sentence2"

    preds = []

    for i in range(test_dataset.shape[0]):
        # tokenize the current sentence pair
        row = test_dataset[i]
        inputs = tokenizer(row[sent1], row[sent2], max_length=512, truncation=True, padding=True, return_tensors='pt')
        inputs.to(device)

        # do inference and get prediction
        outputs = model(**inputs, head_mask=head_mask)
        pred = outputs[0].tolist()[0][0] if task_name == "stsb" else torch.argmax(outputs.logits.softmax(dim=1)).item()
        preds.append(pred)

    # get labels from dataset
    labels = test_dataset['label']

    # calculate performance metric(s)
    metric = evaluate.load("glue", task_name)
    result = metric.compute(predictions=preds, references=labels)

    return result
