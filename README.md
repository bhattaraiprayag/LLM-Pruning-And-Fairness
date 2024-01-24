# Can Pruning Language Models Reduce Their Bias?

This is repository for our team project at the University of Mannheim. This project will explore how a range of pruning methods impact the bias of a Language Model.

To install the necessary packages in a conda environment, follow the instructions in [requirements.txt](requirements.txt). This is currently set up for the evaluation packages.

## Fine-tuning

## Pruning

### Overview
In our project, we focus on exploring the impact of various pruning techniques on the biasness of RoBERTa-base. Pruning, a method to reduce model size and computational load, involves selectively removing parameters (weights), or neurons, from the neural network. We have implemented and experimented with different types of pruning strategies, starting with magnitude-based methods and structure Pruning.

### Structure Pruning:
Structural pruning is a technique for compressing neural networks by removing entire groups of parameters, or filters, based on their structural relationships. 
-**Importance Pruning** A technique for compressing neural networks by removing parameters based on their importance scores. Importance scores are measures of how important each parameter is to the overall performance of the network. 
Variables that we chenge to get the best performance are: **masking threshold ( masking threshold in term of metrics (stop masking when metric < threshold * original metric value)) and **masking amount (mount to heads to masking at each masking step). 
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
- SEAT & WEAT
- StereoSet

As measures of the extrinsic bias of the models we have:
- Bias NLI
- Bias STS - incomplete

#### SEAT and WEAT

This is implemented based on the code published in [BiasBench](https://github.com/McGill-NLP/bias-bench). The data for the tests is stored in [seat](evaluation/data/seat/).

The evaluation can be conducted by running the seatandweat function contained in [seat.py](evaluation/seat.py).

The following files in [utils](evaluation/utils/) are used:
- [utils/seat.py](evaluation/utils/seat.py) - This contains functions and classes from [BiasBench](https://github.com/McGill-NLP/bias-bench) for reading in the test data and carrying out the tests. It also contains newly implemented functions to aggregate the individual test outputs by computing the average absolute effect sizes for the different types of biases.
- [utils/weat.py](evaluation/utils/weat.py) - This comes from [BiasBench](https://github.com/McGill-NLP/bias-bench) and contains functions for carrying out the WEAT tests and producing statistics on the output

The function returns two average absolute effect sizes for gender bias, one for SEAT and one for WEAT.

Additionally, it saves two files in the results folder of a particular experiment run:
- seatandweat_raw.json: contains the p-values and effect sizes of each individual test
- seatandweat_aggregated.json: contains the average absolute effect sizes for the different types if biases (SEAT: gender, race, illness, religion; WEAT: gender, race, illness)

The local run-time is ~10 minutes.

#### StereoSet

This is implemented based on the code published in [ESP - Logic against Bias](https://github.com/luohongyin/esp) rather than the more complex version implmented in [BiasBench](https://github.com/McGill-NLP/bias-bench). The additional functionality offered there was unnecessary for the variants we need.

There is a [utils/stereoset.py](evaluation/utils/stereoset.py) file for the background functions and [stereoset.py](evaluation/stereoset.py) containing the function to be called in the pipeline. There is an output file with all scores saved into the results folder for a particular model and it also returns the basic gender scores for inclusion in the overall results table.

The current settings use the 'intrasentence' setup.

The local run-time is ~15 minutes.

#### Bias NLI

This is implemented based on code published in [On Measuring and Mitigating Biased Inferences of Word Embeddings](https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings/tree/master/word_lists).

The first stage is to generate the templates. This can be accomplished by running:

``` 
python ./evaluation/bias_nli/generate_templates.py --noun --p occupations --h gendered_words --output ./evaluation/data/bias_nli/occupation_gender.csv
```

Other options could be selected for `--p` and `--h`, but we are using this setting initially as they compare occupations with sentences with gendered words. This expands templates into a set of premise-hypothesis pairs and write the result into a CSV file. 
The files used for this are in the [bias_nli folder](evaluation/bias_nli).

There is then a function to produce the scores when it is given a transformers pipeline. It will return a dictionary with net neutral and fraction neutral values.
