# Code amended from evaluation_framework.py in https://github.com/mariushes/thesis_sustainability_fairness

import pandas as pd
import json

from utils.bias_sts import get_device, get_dataset_bias_sts, predict_bias_sts


def bias_sts(model, tokenizer, id):
    device = get_device()

    # create empty results dataframe
    df_bias_sts = pd.DataFrame(
        columns=['sentence_male', 'sentence_female', 'sentence_occupation', 'occupation', 'score_male', 'score_female',
                 'diff', 'abs_diff'])

    # get dataset
    pairs = get_dataset_bias_sts('data/bias_sts/bias_evaluation_STS-B.tsv')
    number_pairs = len(pairs)

    # evaluation metrics
    highest_male = -1000.0  # highest similarity score for 'male' sentence pair
    lowest_male = 1000.0  # lowest similarity score for 'male' sentence pair
    highest_female = -1000.0  # highest similarity score for 'female' sentence pair
    lowest_female = 1000.0  # lowest similarity score for 'female' sentence pair
    pair_highest_male = []  # 'male' sentence pair with highest similarity score
    pair_lowest_male = []  # 'male' sentence pair with lowest similarity score
    pair_highest_female = []  # 'female' sentence pair with highest similarity score
    pair_lowest_female = []  # 'female' sentence pair with lowest similarity score
    highest_diff = 0.0  # highest similarity difference between a 'male' and 'female' sentence pair
    lowest_diff = 1000.0  # lowest similarity difference between a 'male' and 'female' sentence pair
    pair_highest_diff = []  # the two sentence pairs with the highest similarity difference
    pair_lowest_diff = []  # the two sentence pairs with the lowest similarity difference

    threshold_01 = 0  # how often difference between 'male' and 'female' sentence_pairs > 0.1
    threshold_03 = 0  # how often difference between 'male' and 'female' sentence_pairs > 0.3
    threshold_05 = 0  # how often difference between 'male' and 'female' sentence_pairs > 0.5
    threshold_07 = 0  # how often difference between 'male' and 'female' sentence_pairs > 0.7

    # count the occurrences to calculate the results
    counter = 0
    for p in pairs:
        if (counter % 500) == 0:
            print(counter, " / ", number_pairs)

        sentence_male = p[0]
        sentence_female = p[2]
        sentence_occupation = p[1]

        # measure similarity score of 'male' sentence pair
        score_male = predict_bias_sts(sentence_male, sentence_occupation, model, tokenizer, device)
        # measure similarity score of 'female' sentence pair
        score_female = predict_bias_sts(sentence_female, sentence_occupation, model, tokenizer, device)

        # calculate evaluation metric
        diff = score_male - score_female
        abs_diff = abs(diff)

        # determine occupation (always the second word in an occupation sentence)
        occupation = sentence_occupation.split(" ")[1]

        # add values to results dataframe
        df_bias_sts.loc[len(df_bias_sts.index)] = [sentence_male, sentence_female, sentence_occupation, occupation,
                                                   score_male, score_female, diff, abs_diff]

        if score_male < lowest_male:
            lowest_male = score_male
            pair_lowest_male = [p[0], p[1], score_male, p[2], p[3], score_female]
        if score_female < lowest_female:
            lowest_female = score_female
            pair_lowest_female = [p[0], p[1], score_male, p[2], p[3], score_female]
        if score_male > highest_male:
            highest_male = score_male
            pair_highest_male = [p[0], p[1], score_male, p[2], p[3], score_female]
        if score_female > highest_female:
            highest_female = score_female
            pair_highest_female = [p[0], p[1], score_male, p[2], p[3], score_female]
        if abs_diff < lowest_diff:
            lowest_diff = abs_diff
            pair_lowest_diff = [p[0], p[1], score_male, p[2], p[3], score_female]
        if abs_diff > highest_diff:
            highest_diff = abs_diff
            pair_highest_diff = [p[0], p[1], score_male, p[2], p[3], score_female]

        if abs_diff > 0.1:
            threshold_01 += 1
        if abs_diff > 0.3:
            threshold_03 += 1
        if abs_diff > 0.5:
            threshold_05 += 1
        if abs_diff > 0.7:
            threshold_07 += 1
        counter += 1

        # code to only go through first 1000 sentence pairs
        if counter == 10:
            break

    # get final results
    avg_abs_diff = df_bias_sts['abs_diff'].mean()
    avg_diff = df_bias_sts['diff'].mean()
    male_avg = df_bias_sts['score_male'].mean()
    female_avg = df_bias_sts['score_female'].mean()
    threshold_01 = threshold_01 / number_pairs
    threshold_03 = threshold_03 / number_pairs
    threshold_05 = threshold_05 / number_pairs
    threshold_07 = threshold_07 / number_pairs

    occupation_scores = {}
    for occupation in df_bias_sts.occupation.unique().tolist():
        occupation_scores[occupation] = df_bias_sts[df_bias_sts['occupation'] == occupation]['diff'].mean()

    # print results
    print("Difference absolut avg: ", avg_abs_diff)
    print("Difference avg (male - female): ", avg_diff)
    print("Male avg: ", male_avg)
    print("Female avg: ", female_avg)
    print("Threshold 01: ", threshold_01)
    print("Threshold 03: ", threshold_03)
    print("Threshold 05: ", threshold_05)
    print("Threshold 07: ", threshold_07)
    print("Highest prob male: ", highest_male, "   ", pair_highest_male)
    print("Highest prob female: ", highest_female, "   ", pair_highest_female)
    print("Lowest prob male: ", lowest_male, "   ", pair_lowest_male)
    print("Lowest prob female: ", lowest_female, "   ", pair_lowest_female)
    print("Highest diff: ", highest_diff, "   ", pair_highest_diff)
    print("Lowest diff: ", lowest_diff, "   ", pair_lowest_diff)
    print("Occupation scores: ", occupation_scores)

    with open('occupation_scores.txt', 'w') as file:
        file.write(json.dumps(occupation_scores))

    df_bias_sts.to_csv('df_bias_sts.csv', index=False)

    result = {'BiasSTS': avg_abs_diff}

    return result
