import json
import utils

def main():
    new_df = utils.get_id_info('results/results.csv')

    Gender, Profession, Race, Religion = ([] for i in range(4))

    for i in new_df['ID']:
        with open(f'results/run{i}/stereoset.json', 'r') as f:
            json_file = json.load(f)
            Gender.append(json_file['gender']['SS'])
            Profession.append(json_file['profession']['SS'])
            Race.append(json_file['race']['SS'])
            Religion.append(json_file['religion']['SS'])

    new_df = new_df.assign(gender = Gender,
                           profession = Profession,
                           race = Race,
                           religion = Religion)

    new_df.to_csv('results/results_stereoset.csv', index=False)

if __name__ == "__main__":
    main()