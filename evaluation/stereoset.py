# Code amended from https://github.com/luohongyin/ESP/blob/main/eval_stereo.py

import json
import os

import evaluation.utils.stereoset as ss

def stereoset(model, tokenizer, exp_id):

    # Setup
    eval_split = 'intrasentence' # More commonly used, but can also get intersentence
    cls_head = int(0) # Can be 0, 1 or 2 for entailment, neutral or contradiction
    eval_mode = 'score'

    # Get data
    sent_list, data = ss.load_data('evaluation/data/StereoSet/test.json', eval_split)

    # Get output values
    score_board, pred_board = ss.cls_evaluate(tokenizer, model, cls_head, sent_list, batch_size=4)

    # Generate scores
    bias_type_list = ['gender', 'profession', 'race', 'religion', 'overall']
    overall = {}

    for bias_type in bias_type_list:
        working = {}
        lm_score, stereo_score, icat_score = ss.calculate_icat(
            score_board if eval_mode == 'score' else pred_board,
            sent_list, data,
            bias_type=bias_type, input_type=eval_mode
        )

        working['LM'] = lm_score
        working['SS'] = stereo_score
        working['iCAT'] = icat_score
        overall[bias_type] = working

    # Save whole output file
    os.makedirs(f'results/run{exp_id}', exist_ok=True)
    with open(f'results/run{exp_id}/stereoset.json', 'w') as f:
        json.dump(overall, f, indent=2)

    # Return desired values for table
    results = {}
    results['stereoset_LM_gender']=overall['gender']['LM']
    results['stereoset_SS_gender'] = overall['gender']['SS']

    return results

