import json
import pandas as pd

def get_id_info(filepath):
    # Read in current results data frame
    results_df = pd.read_csv(filepath)
    # Keep just the columns of interest
    new_df = results_df[['ID', 'date', 'device', 'seed', 'task', 'pruning_method', 'sparsity_level', 'temperature']]
    return new_df

def main():
    new_df = get_id_info('results/results.csv')

    SEAT_gender, SEAT_race, SEAT_illness, SEAT_religion, WEAT_gender, WEAT_race, WEAT_illness = ([] for i in range(7))

    for i in [1]: #new_df['ID']:
        with open(f'results/run{i}/seatandweat_aggregated.json', 'r') as f:
            json_file = json.load(f)
            SEAT_gender.append(json_file['SEAT_gender'])
            SEAT_race.append(json_file['SEAT_race'])
            SEAT_illness.append(json_file['SEAT_illness'])
            SEAT_religion.append(json_file['SEAT_religion'])
            WEAT_gender.append(json_file['WEAT_gender'])
            WEAT_race.append(json_file['WEAT_race'])
            WEAT_illness.append(json_file['WEAT_illness'])

    new_df = new_df.assign(seat_gender = SEAT_gender,
                           seat_race = SEAT_race,
                           seat_illness = SEAT_illness,
                           seat_religion = SEAT_religion,
                           weat_gender = WEAT_gender,
                           weat_race = WEAT_race,
                           weat_illness = WEAT_illness)

    new_df.to_csv('results/results_seat_weat.csv', index=False)

if __name__ == "__main__":
    main()