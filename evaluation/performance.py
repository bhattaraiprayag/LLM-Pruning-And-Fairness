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

def load_eval_dataset(task):    # ONLY WORKS WITH split='validation'
    """Loads the evaluation dataset based on the specified task."""
    if task == 'mnli':
        return (
            load_dataset('glue', 'mnli', split='validation_matched'),
            load_dataset('glue', 'mnli', split='validation_mismatched')
        )
    elif task == 'stsb':
        return load_dataset('glue', 'stsb', split='validation'),
    else:
        raise ValueError(f'No evaluation dataset found for task {task}')


def evaluate_metrics(model, tokenizer, task, eval_datasets):
    """Evaluates task-specific metrics and returns results."""
    results_dict = {}
    if task == 'mnli':
        eval_matched, eval_mismatched = eval_datasets
        mnli_matched = evaluate_model(model, tokenizer, task, eval_matched)
        mnli_mismatched = evaluate_model(model, tokenizer, task, eval_mismatched)
        results_dict['Matched Acc'], results_dict['Mismatched Acc'] = mnli_matched['eval_accuracy'], mnli_mismatched['eval_accuracy']
    elif task == 'stsb':
        eval_dataset = eval_datasets[0]
        eval_results = evaluate_model(model, tokenizer, task, eval_dataset)
        results_dict['Spearmanr'], results_dict['Pearson'] = eval_results['eval_spearmanr'], eval_results['eval_pearson']
    else:
        raise ValueError(f'No evaluation metrics found for task {task}')
    # return print(f"Task: {task.upper()} | {results_dict}")
    return results_dict


def evaluate_model(model, tokenizer, task_name, eval_dataset):
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
    
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # define compute metrics function
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if task_name == "stsb" else np.argmax(preds, axis=1)
        metric = evaluate.load("glue", task_name)
        return metric.compute(predictions=preds, references=p.label_ids)

    # define trainer
    training_args = TrainingArguments(
        output_dir="./results",  # Temporary directory for storing evaluation results
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