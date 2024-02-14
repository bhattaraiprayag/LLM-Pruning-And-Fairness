import evaluate
import numpy as np

from transformers import Trainer, TrainingArguments, EvalPrediction
from datasets import load_dataset


# def load_eval_dataset(task): FOR LOADING LOCAL FILES; local test set doesn't have labels | WORK IN PROGRESS
#     """Loads the .tsv test datasets based on the specified task, from local folder"""
#     mnli_path = "/training/glue_data/MNLI/"
#     stsb_path = "/training/glue_data/STS-B/"
#     if task == 'mnli':
#         return (
#             load_dataset("glue", "mnli", data_dir=mnli_path, split='test_matched'),
#             load_dataset("glue", "mnli", data_dir=mnli_path, split='test_mismatched')
#         )
#     elif task == 'stsb':
#         return load_dataset("glue", "stsb", data_dir=stsb_path, split='test')
#     else:
#         raise ValueError(f'No dataset found for task {task}')

def load_eval_dataset(task, model_no):    # ONLY WORKS WITH split='validation'
    """Loads the evaluation dataset based on the specified task."""
    if task == 'mnli':
        if model_no == 1: # uses original split of the data
            return (
                load_dataset('glue', 'mnli', split='validation_matched[-50%:]'),
                load_dataset('glue', 'mnli', split='validation_mismatched[-50%:]')
            )
        else: # uses shuffled split based on seed
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
            return(
                test_valid['train'],
                full_dataset[2]
            )

    elif task == 'stsb':
        if model_no == 1: # uses original split of the data
            return load_dataset('glue', 'stsb', split='validation[-50%:]')
        else: # uses shuffled split based on seed
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


def evaluate_metrics(model, tokenizer, task, eval_datasets, exp_id):
    """Evaluates task-specific metrics and returns results."""
    results_dict = {}
    if task == 'mnli':
        eval_matched, eval_mismatched = eval_datasets
        mnli_matched = evaluate_model(model, tokenizer, task, eval_matched, exp_id)
        mnli_mismatched = evaluate_model(model, tokenizer, task, eval_mismatched, exp_id)
        results_dict['Matched Acc'], results_dict['Mismatched Acc'] = mnli_matched['eval_accuracy'], mnli_mismatched['eval_accuracy']
    elif task == 'stsb':
        eval_results = evaluate_model(model, tokenizer, task, eval_datasets, exp_id)
        results_dict['Spearmanr'], results_dict['Pearson'] = eval_results['eval_spearmanr'], eval_results['eval_pearson']
    else:
        raise ValueError(f'No evaluation metrics found for task {task}')
    # return print(f"Task: {task.upper()} | {results_dict}")
    return results_dict


def evaluate_model(model, tokenizer, task_name, eval_datasets, exp_id):
    # tokenization
    def preprocess_function(examples, task_name=task_name):
        if task_name == "mnli":
            # premise and hypothesis
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding=True)
        elif task_name == "stsb":
            # sentence pairs
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=True)
        else:
            raise ValueError(f"Task {task_name} not yet supported")
    
    eval_dataset = eval_datasets.map(preprocess_function, batched=True)

    # define compute metrics function
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if task_name == "stsb" else np.argmax(preds, axis=1)
        metric = evaluate.load("glue", task_name)
        return metric.compute(predictions=preds, references=p.label_ids)

    # define trainer
    training_args = TrainingArguments(
        output_dir=f"./results/run{exp_id}",  # Temporary directory for storing evaluation results
        do_train=False,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    return trainer.evaluate()