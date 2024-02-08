import numpy as np


def predict_bias_sts(df, model_pipe):
    """
    Args:
        df: data frame containing sentence pairs (sentence_gender and sentence_occupation)
        model_pipe: huggingface pipeline

    Returns: array of similarity scores for each input sentence pair
    """
    # Get all input sentence pairs in a list of tuples
    inputs_df = df[['sentence_gender', 'sentence_occupation']]
    inputs = inputs_df.to_records(index=False).tolist()

    # Make predictions with model and calculate similarity scores (range 0 to 5)
    pred = model_pipe(inputs)
    scores = np.array([i[0]['score'] for i in pred]) * 5

    return scores
