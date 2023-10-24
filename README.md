# Can Pruning Language Models Reduce Their Bias?

This is repository for our team project at the University of Mannheim. This project will explore how a range of pruning methods impact the bias of a Language Model.

## Fine-tuning

## Pruning

## Evaluation

### Model evaluation

### Bias evaluation

We have implemented a number of measures of bias.

As measures of the intrinsic bias of the models we have:
- SEAT and WEAT

There are some shared scripts within the [utils](evaluation/utils/) folder:
- [models.py](evaluation/utils/models.py) - This comes from https://github.com/McGill-NLP/bias-bench and handles loading a model
- [experiment_id.py](evaluation/utils/experiment_id.py) - This comes from https://github.com/McGill-NLP/bias-bench and handles the naming of the output file

#### SEAT and WEAT

This is implemented based on the code published in https://github.com/McGill-NLP/bias-bench. The data for the tests is stored in [seat](evaluation/data/seat/).

The following files in [utils](evaluation/utils/) are used:
- [seat.py](evaluation/utils/seat.py) - This comes from https://github.com/McGill-NLP/bias-bench and contains functions for reading in the test data and carrying out the tests
- [weat.py](evaluation/utils/weat.py) - This comes from https://github.com/McGill-NLP/bias-bench and contains functions for carrying out the WEAT tests and producing statistics on the output

The final evaluation can be conducted by running [seat.py](evaluation/seat.py). There are optional variables for amending the filepath to the model or other variables. An example of this would be to run the following in the terminal:
```python ./evaluation/seat.py --model_name_or_path <model_path>```

