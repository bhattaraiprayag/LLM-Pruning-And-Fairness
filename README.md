# Can Pruning Language Models Reduce Their Bias?

This is repository for our team project at the University of Mannheim. This project will explore how a range of pruning methods impact the bias of a Language Model.

To install the necessary packages in a conda environment, follow the instructions in [requirements.txt](requirements.txt). This is currently set up for the evaluation packages.

## Fine-tuning

## Pruning

### Overview
In our project, we focus on exploring the impact of various pruning techniques on the biasness of RoBERTa-base. Pruning, a method to reduce model size and computational load, involves selectively removing parameters (weights), or neurons, from the neural network. We have implemented and experimented with different types of pruning strategies, starting with magnitude-based methods.

### Magnitude Pruning:
Magnitude pruning is a method for reducing the size and complexity of an LLM/neural networks by selectively removing parameters (weights) based on their magnitudes. Our **MagnitudePrunerOneShot** class, defined in *magnitude_pruner.py*, is central to our pruning strategy. This class offers three distinct methods of magnitude-based pruning:
- **L1-Unstructured**: This global pruning strategy removes weights across the entire network based on their L1-norm magnitude. Can be used with *--pruning_method l1-unstructured*.
- **L1-Unstructured (Linear)**: Targets only the linear layers of the model, pruning weights based on their L1-norm. Can be used with *--pruning_method l1-unstructured-linear*.
- **L1-Unstructured Invert**: (To be implemented) Aimed at exploring inverted criteria for pruning. IN PROGRESS.

We ensure that each pruning process begins with a consistent state by setting a seed for reproducibility. Post-pruning, we evaluate and report the sparsity levels of the model to understand the extent of weight reduction.

## Evaluation

### Performance evaluation
To gauge the performance of our pruned models, we turn to our benchmark tasks: the Multi-Genre Natural Language Inference (MNLI) and the Semantic Textual Similarity Benchmark (STS-B). These tasks allow us to assess the model's understanding of language and its ability to capture semantic relationships, respectively.

Our performance.py script encapsulates the evaluation pipeline:
* Dataset Loading: We load the validation datasets for MNLI and STS-B, accommodating both matched and mismatched scenarios for MNLI.
* Evaluation Functionality: The evaluate_metrics function orchestrates the evaluation process. It leverages the evaluate_model function to perform task-specific assessments, returning a dictionary of key performance metrics.
* Metrics Computation:
  * For MNLI: We report accuracy for both matched and mismatched datasets.
  * For STS-B: We measure performance using Spearman’s rank correlation coefficient and Pearson correlation coefficient.
The evaluation process involves tokenizing the datasets and feeding them through the model using Hugging Face's Trainer API. We then compute the metrics using the predictions and labels.

### Bias evaluation

We have implemented a number of measures of bias.

As measures of the intrinsic bias of the models we have:
- SEAT and WEAT

There are some shared scripts within the [utils](evaluation/utils/) folder:
- [models.py](evaluation/utils/models.py) - This comes from [BiasBench](https://github.com/McGill-NLP/bias-bench) and handles loading a model
- [model_utils.py](evaluation/utils/model_utils.py) - This comes from [BiasBench](https://github.com/McGill-NLP/bias-bench) and handles some checks about the model type.
- [experiment_id.py](evaluation/utils/experiment_id.py) - This comes from [BiasBench](https://github.com/McGill-NLP/bias-bench) and handles the naming of the output file

#### SEAT and WEAT

This is implemented based on the code published in [BiasBench](https://github.com/McGill-NLP/bias-bench). The data for the tests is stored in [seat](evaluation/data/seat/).

The following files in [utils](evaluation/utils/) are used:
- [seat.py](evaluation/utils/seat.py) - This comes from [BiasBench](https://github.com/McGill-NLP/bias-bench) and contains functions for reading in the test data and carrying out the tests
- [weat.py](evaluation/utils/weat.py) - This comes from [BiasBench](https://github.com/McGill-NLP/bias-bench) and contains functions for carrying out the WEAT tests and producing statistics on the output

The final evaluation can be conducted by running [seat.py](evaluation/seat.py). There are optional variables for amending the filepath to the model or other variables. An example of this would be to run the following in the terminal:

```
python ./evaluation/seat.py --model_name_or_path <model_path>
```

The local run-time is ~10 minutes.

#### StereoSet

This is implemented based on the code published in [BiasBench](https://github.com/McGill-NLP/bias-bench). The data for the tests is stored in [stereoset](evaluation/data/stereoset/).

The following files in [utils](evaluation/utils/) are used:
- [stereoset.py](evaluation/utils/stereoset.py) - This comes from [BiasBench](https://github.com/McGill-NLP/bias-bench) and contains functions for reading in the test data and carrying out the tests
- [stereoset_dataloader.py](evaluation/utils/stereoset_dataloader.py) - This comes from [BiasBench](https://github.com/McGill-NLP/bias-bench) and handles loading the data correctly for the tests

The evaluation can be conducted by running [stereoset.py](evaluation/stereoset.py) for the specific model, in a similar way to in [seat](#seat-and-weat). The local run-time is ~1 hour.

The output files must then be run with a summariser from [stereoset2.py](evaluation/stereoset2.py), to give actual values for how correct the model was. This can be done with a single call for all files:

```
python ./evaluation/stereoset2.py --predictions_dir "./evaluation/results/stereoset/"
```
