# Code amended from https://github.com/luohongyin/ESP/blob/main/eval_stereo.py

import json

import evaluation.utils.stereoset as ss

def stereoset(model, tokenizer, exp_id):

    eval_split = 'intrasentence' # More commonly used, but can also get intersentence
    cls_head = int(0) # Can be 0, 1 or 2 for entailment, neutral or contradiction
    eval_mode = 'score'

    sent_list, data = ss.load_data('evaluation/data/StereoSet/test.json', eval_split)

    score_board, pred_board = ss.cls_evaluate(tokenizer, model, cls_head, sent_list, batch_size=4)

    bias_type_list = ['gender', 'profession', 'race', 'religion', 'overall']

    print('=========================================================')
    print(f'============== Exp of cls_head: {cls_head} ==============')
    print('=========================================================')

    overall = {}

    for bias_type in bias_type_list:
        working = {}
        print(f'\n------- {bias_type} -------')
        lm_score, stereo_score, icat_score = ss.calculate_icat(
            score_board if eval_mode == 'score' else pred_board,
            sent_list, data,
            bias_type=bias_type, input_type=eval_mode
        )

        print(f'LM score = {lm_score}')
        print(f'SS score = {stereo_score}')
        print(f'iCat score = {icat_score}')
        working['LM'] = lm_score
        working['SS'] = stereo_score
        working['iCAT'] = icat_score
        overall[bias_type] = working

    # Save whole output file
    with open(f'results/{exp_id}/stereoset.json', 'w') as f:
        json.dump(overall, f, indent=2)

    # Return desired values for table
    results = {}
    results['SS_LM_gender']=overall['gender']['LM']
    results['SS_SS_gender'] = overall['gender']['SS']

    return results

