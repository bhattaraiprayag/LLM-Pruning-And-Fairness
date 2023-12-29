import transformers

class Model:
    def __new__(self, model_path):
        return transformers.RobertaForSequenceClassification.from_pretrained(model_path, use_safetensors=True, local_files_only=True)

