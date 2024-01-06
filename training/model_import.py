import transformers

model = transformers.RobertaForSequenceClassification.from_pretrained("final_models/MNLI/", use_safetensors=True, local_files_only=True)

print(model)