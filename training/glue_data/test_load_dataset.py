from datasets import load_dataset, DatasetDict

# STS-B
raw_datasets_stsb = load_dataset(
    "glue",
    "stsb",
    split=['train', 'validation[:50%]', 'validation[-50%:]']
)
# turn datasets into a DatasetDict
raw_datasets_stsb = DatasetDict({'train': raw_datasets_stsb[0], 'validation': raw_datasets_stsb[1], 'test': raw_datasets_stsb[2]})

#print(raw_datasets_stsb)

# MNLI
raw_datasets_mnli = load_dataset(
    "glue",
    "mnli",
    split=['train', 'validation_matched[:50%]', 'validation_mismatched[:50%]', 'validation_matched[-50%:]',
           'validation_mismatched[-50%:]']
)
# turn datasets into a DatasetDict
raw_datasets_mnli = DatasetDict({'train': raw_datasets_mnli[0],
                                 'validation_matched': raw_datasets_mnli[1],
                                 'validation_mismatched': raw_datasets_mnli[2],
                                 'test_matched': raw_datasets_mnli[3],
                                 'test_mismatched': raw_datasets_mnli[4]
                                 })

#print(raw_datasets_mnli)
