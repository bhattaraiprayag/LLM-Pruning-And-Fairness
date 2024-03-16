# Can Pruning Language Models Reduce Their Bias?

This is repository for our team project at the University of Mannheim. This project will explore how a range of pruning methods impact the bias of a Language Model.

## Fine-tuning

Fine-tuning for our two tasks, MNLI and STS-B, is done using the [run_glue.py](training/run_glue.py) script made available by [huggingface](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py). For each task three models are fine-tuned, each using a different train-validation-test split.
The data files for each split can be saved by running the script [save_data.py](training/glue_data/save_data.py).

## Pruning

### Overview
In our project, we focus on exploring the impact of various pruning techniques on the biasness of RoBERTa-base. Pruning, a method to reduce model size and computational load, involves selectively removing parameters (weights), or neurons, from the neural network. We have implemented and experimented with different types of pruning strategies, starting with magnitude-based methods and structure Pruning.

### Structured Pruning:
Structured pruning is implemented in [structure_pruning.py](pruning/structure_pruning.py) and is based on the code published in [When BERT Plays the Lottery, All Tickets Are Winning](https://github.com/sai-prasanna/bert-experiments). 

The variable that can be changed is the masking threshold, which defines the performance threshold for stopping masking. For performance evaluation within the the corresponding validation set of the used model is utilized. The masking amount (the fraction of heads to mask in each iteration) is kept constant at 0.1.


### Magnitude Pruning:
Magnitude pruning is implemented through the **MagnitudePrunerOneShot** class, defined in [pruning/magnitude_pruner.py](magnitude_pruner.py). This class offers three distinct methods of magnitude-based pruning:
* L1-Unstructured: This global pruning strategy removes weights across the entire network based on their L1-norm magnitude. Can be used with *--pruning_method l1-unstructured*.
* L1-Unstructured (Linear): Targets only the linear layers of the model, pruning weights based on their L1-norm. Can be used with *--pruning_method l1-unstructured-linear*.
* Random-Unstructured: Randomly prunes weights to the specified sparsity

We ensure that each pruning process begins with a consistent state by setting a seed for reproducibility. Post-pruning, we evaluate and report the sparsity levels of the model to understand the extent of weight reduction.

## Evaluation

### Performance evaluation
To gauge the performance of our pruned models, we turn to our benchmark tasks: the Multi-Genre Natural Language Inference (MNLI) and the Semantic Textual Similarity Benchmark (STS-B). These tasks allow us to assess the model's understanding of language and its ability to capture semantic relationships, respectively.

Our [performance.py](evaluation/performance.py) script encapsulates the evaluation pipeline:
* Dataset Loading: We load the validation datasets for MNLI and STS-B, accommodating both matched and mismatched scenarios for MNLI.
* Evaluation Functionality: The evaluate_metrics function orchestrates the evaluation process. It leverages the evaluate_model function to perform task-specific assessments, returning a dictionary of key performance metrics.
* Metrics Computation:
  - For MNLI: We report accuracy for both matched and mismatched datasets.
  - For STS-B: We measure performance using Spearman’s rank correlation coefficient and Pearson correlation coefficient.
The evaluation process involves tokenizing the datasets and feeding them through the model using Hugging Face's Trainer API. We then compute the metrics using the predictions and labels.

[performance_check.py](performance_check.py) allows for testing these metrics over a range of pruning depths. Whilst the focus of the results overall is the model biases, if the model performance is unusable then the results would be useless. This script requires one of the fine-tuned models and then prunes to a range of sparsity levels and saved the results.

### Bias evaluation

We have implemented a number of measures of bias.

As measures of the intrinsic bias of the models we have:
* SEAT & WEAT
* StereoSet

As measures of the extrinsic bias of the models we have:
* Bias NLI
* Bias STS

#### SEAT and WEAT

This is implemented based on the code published in [BiasBench](https://github.com/McGill-NLP/bias-bench). The data for the tests is stored in [seat](evaluation/data/seat/).

The evaluation can be conducted by running the seatandweat function contained in [seat.py](evaluation/seat.py).

The following files in [utils](evaluation/utils/) are used:
* [utils/seat.py](evaluation/utils/seat.py): This contains functions and classes from [BiasBench](https://github.com/McGill-NLP/bias-bench) for reading in the test data and carrying out the tests. It also contains newly implemented functions to aggregate the individual test outputs by computing the average absolute effect sizes for the different types of biases.
* [utils/weat.py](evaluation/utils/weat.py): This comes from [BiasBench](https://github.com/McGill-NLP/bias-bench) and contains functions for carrying out the WEAT tests and producing statistics on the output

The function returns two average absolute effect sizes for gender bias, one for SEAT and one for WEAT.

Additionally, it saves two files in the results folder of a particular experiment run:
* seatandweat_raw.json: contains the p-values and effect sizes of each individual test
* seatandweat_aggregated.json: contains the average absolute effect sizes for the different types if biases (SEAT: gender, race, illness, religion; WEAT: gender, race, illness)

The local run-time is ~10 minutes.

#### StereoSet

This is implemented based on the code published in [ESP - Logic against Bias](https://github.com/luohongyin/esp) rather than the more complex version implmented in [BiasBench](https://github.com/McGill-NLP/bias-bench). The additional functionality offered there was unnecessary for the variants we need.

There is a [utils/stereoset.py](evaluation/utils/stereoset.py) file for the background functions and [stereoset.py](evaluation/stereoset.py) containing the function to be called in the pipeline. There is an output file with all scores saved into the results folder for a particular model and it also returns the basic gender scores for inclusion in the overall results table.

The current settings use the 'intrasentence' setup.

The local run-time is ~15 minutes.

#### Bias-NLI

This is implemented based on code published in [On Measuring and Mitigating Biased Inferences of Word Embeddings](https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings/tree/master/word_lists).

The first stage is to generate the templates. This can be accomplished by running:

``` 
python ./evaluation/bias_nli/generate_templates.py --noun --p occupations --h gendered_words --output ./evaluation/data/bias_nli/occupation_gender.csv
```

Other options could be selected for `--p` and `--h`, but we are using this setting initially as they compare occupations with sentences with gendered words. This expands templates into a set of premise-hypothesis pairs and write the result into a CSV file. 
The files used for this are in the [bias_nli folder](evaluation/bias_nli).

There is then a function to produce the scores when it is given a model and tokenizer. It will return a dictionary with net neutral and fraction neutral values. An output file is also saved with all predictions, NOTE: 'entailed' and 'contradicted' are incorrectly switched in this additional file.

#### Bias-STS

This is implemented based on the code published in [Sustainability and Fairness in Pretrained Language Models: Analysis and Mitigation of Bias when Distilling BERT](https://github.com/mariushes/thesis_sustainability_fairness/blob/master/evaluation_framework.py) using the corresponding [dataset](https://github.com/mariushes/thesis_sustainability_fairness/blob/master/datasets/bias_evaluation_STS-B.tsv).

The final function that is called in the pipeline is contained in [bias_sts.py](evaluation/bias_sts.py). It returns the absolute average difference, which is the evaluation score of the method. It also saves a json file in the results folder of the run that contains a dict with the average differences for each occupation and a csv file with the scores for each sentence pair.

## Report

The graphs for the report are produced using [plots.R](report/plots.R) in R. The base folder where the repo is stored must be manually included in the code. They are saved to the figures folder.

The tables for the report based on data are produced using the functions in [tables.py](report/tables.py). They are saved to the tables folder in a latex folder.

[seat_weat.py](report/seat_weat.py) and [stereoset.py](report/stereoset.py) are both able to gather the extra data saved for each of those bias measures in the individual folders for each run. They are combined into a single csv. This data can then be analysed further.
