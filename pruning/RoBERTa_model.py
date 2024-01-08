import transformers

#If you are working with RoBERTa instead of BERT, you need to use the appropriate configuration class for RoBERTa. In this case, it would be RobertaConfig. Here's how you can modify your code to use RoBERTa's configuration:


from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
model = transformers.RobertaForSequenceClassification.from_pretrained("models/MNLI_model/", use_safetensors=True, local_files_only=True)

print(model)