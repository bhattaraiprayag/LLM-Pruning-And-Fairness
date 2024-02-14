import json
import utils

def main():
    new_df = utils.get_id_info('results/results.csv')

    SEAT_gender, SEAT_race, SEAT_illness, SEAT_religion, WEAT_gender, WEAT_race, WEAT_illness = ([] for i in range(7))

    for i in new_df['ID']:
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