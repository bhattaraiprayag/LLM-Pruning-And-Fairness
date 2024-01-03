# Adapted from:
# - https://github.com/Stellakats/Master-thesis-gender-bias/blob/main/dataloader/create_bias_dataset.py
# - https://github.com/Stellakats/Master-thesis-gender-bias/blob/main/utils/misc.py#L33

import os
import csv
import pandas as pd
from tqdm import tqdm


# Receives the STS-B test set and creates gender-occupation datasets


class CreateGenderStsb():
    def __init__(self, data_dir=None, occupation=None):
        self.data_dir = data_dir
        self.occupation = occupation

    def create_gendered_dataframes(self):
        """
        Creates one dataframe for woman and one for man.
        Each dataset consists of 173 pairs of sentences:
        each pair contains one gendered sentence and one that contains the occupation.
        """
        df = self.create_dataframe()

        # create men and women dataframes
        women = df[df['gender'] == 'woman']
        men = df[df['gender'] == 'man']

        # create copies of men and women dataframes
        men2 = women.copy()
        women2 = men.copy()

        # transform the copies to opposite gender ones
        men2['sentence1'] = men2['sentence1'].apply(lambda x: self.replace_with(x, 'man'))
        women2['sentence1'] = women2['sentence1'].apply(lambda x: self.replace_with(x, 'woman'))

        # concatenate dataframes of same gender
        women_df = pd.concat([women, women2])
        men_df = pd.concat([men, men2])

        # keep only sentence pairs
        men_df = men_df[['sentence1', 'occupation']]
        men_df = men_df.sort_index(axis=0)
        women_df = women_df[['sentence1', 'occupation']]
        women_df = women_df.sort_index(axis=0)

        return women_df, men_df

    def create_dataframe(self):
        # infers on test set
        test_path = os.path.join(self.data_dir, 'sts-test.csv')
        df = pd.read_csv(test_path, delimiter='\t', header=None, quoting=csv.QUOTE_NONE, usecols=range(7))
        print(f'len of test set: {df.shape}')
        df = df.rename(columns={5: "sentence1", 6: "sentence2"})
        df = df[['sentence1', 'sentence2']]
        # if sentence1 start with 'A man' or 'A woman' and lacks gender-pronouns, replace 'man' or 'woman'
        # with occupation in a new column:
        df['occupation'] = df['sentence1'].apply(
            lambda x: self.replace_with(x, self.occupation) if self.is_gendered(x) else False)
        # indicate the gender of a sentence
        df['gender'] = df['sentence1'].apply(lambda x: 'man' if x[:5] == 'A man' else 'woman')
        # keep only those sentences which start with 'a man' or 'a woman' and lack gender-pronouns
        df = df[df.occupation != False]
        return df

    def is_gendered(self, x):
        if x[:5] == 'A man':
            pronouns = ['his', 'he', 'him']
            if any([word in pronouns for word in [i.split('.')[0] for i in x.split(' ')]]):
                return False
            else:
                return True
        if x[:7] == 'A woman':
            pronouns = ['her', 'she', 'hers']
            if any([word in pronouns for word in [i.split('.')[0] for i in x.split(' ')]]):
                return False
            else:
                return True

    def replace_with(self, x, occupation):
        vowels = ['a', 'e', 'i', 'o', 'u']
        if x[:5] == 'A man':
            if occupation[0] in vowels:
                x = x.replace('A man ', 'An ' + occupation + ' ')
            else:
                x = x.replace('A man ', 'A ' + occupation + ' ')
        if x[:7] == 'A woman':
            if occupation[0] in vowels:
                x = x.replace('A woman ', 'An ' + occupation + ' ')
            else:
                x = x.replace('A woman ', 'A ' + occupation + ' ')
        return x


occupations = ['technician', 'accountant', 'supervisor', 'engineer', 'worker', 'educator', 'clerk', 'counselor',
               'inspector', 'mechanic', 'manager', 'therapist', 'administrator', 'salesperson', 'receptionist',
               'librarian', 'advisor', 'pharmacist', 'janitor', 'psychologist', 'physician', 'carpenter', 'nurse',
               'investigator', 'bartender', 'specialist', 'electrician', 'officer', 'pathologist', 'teacher', 'lawyer',
               'planner', 'practitioner', 'plumber', 'instructor', 'surgeon', 'veterinarian', 'paramedic', 'examiner',
               'chemist', 'machinist', 'appraiser', 'nutritionist', 'architect', 'hairdresser', 'baker', 'programmer',
               'paralegal', 'hygienist', 'scientist', 'dispatcher', 'cashier', 'auditor', 'dietitian', 'painter',
               'broker', 'chef', 'doctor', 'firefighter', 'secretary']


def create_df():
    """
    creates two dataframes (one for men and one for women) containing all 60 occupations
    """

    all_occupations_men = pd.DataFrame(columns=['sentence1', 'occupation'])
    all_occupations_women = pd.DataFrame(columns=['sentence1', 'occupation'])

    for i, occupation in enumerate(tqdm(occupations)):
        print(f'occupation {i + 1}/{len(occupations)}...')
        dataset_creator = CreateGenderStsb(data_dir='', occupation=occupation)
        women_df, men_df = dataset_creator.create_gendered_dataframes()
        all_occupations_men = pd.concat([all_occupations_men, men_df])
        all_occupations_women = pd.concat([all_occupations_women, women_df])

    all_occupations_women.to_csv('../data/bias_sts/women.csv', index=False)
    all_occupations_men.to_csv('../data/bias_sts/men.csv', index=False)


create_df()
