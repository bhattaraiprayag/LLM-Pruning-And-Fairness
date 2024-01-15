import transformers

model = transformers.RobertaForSequenceClassification.from_pretrained("models/MNLI_model/", use_safetensors=True, local_files_only=True)

print(model)