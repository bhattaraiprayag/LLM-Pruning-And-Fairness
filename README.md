# Can Pruning Language Models Reduce Their Bias?

This is repository for our team project at the University of Mannheim. This project will explore how a range of pruning methods impact the bias of a Language Model.

To install the necessary packages in a conda environment, follow the instructions in [requirements.txt](requirements.txt). This is currently set up for the evaluation packages.

## Fine-tuning

## Pruning

## Evaluation

### Model evaluation

### Bias evaluation

We have implemented a number of measures of bias.

As measures of the intrinsic bias of the models we have:
- SEAT and WEAT
- StereoSet

As measures of the extrinsic bias of the models we have:
- Bias NLI - Incomplete

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

#### Bias NLI

This is implemented based on code published in [On Measuring and Mitigating Biased Inferences of Word Embeddings](https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings/tree/master/word_lists).

The first stage is to generate the templates. This can be accomplished by running:

``` 
python ./evaluation/bias_nli/generate_templates.py --noun --p occupations --h gendered_words --output ./evaluation/data/bias_nli/occupation_gender.csv
```

Other options could be selected for `--p` and `--h`, but we are using this setting initially as they compare occupations with sentences with gendered words. This expands templates into a set of premise-hypothesis pairs and write the result into a CSV file.