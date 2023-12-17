from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model = AutoModelForSeq2SeqLM.from_pretrained('models/flan-t5-small').to('mps')
tokenizer = AutoTokenizer.from_pretrained('models/flan-t5-small')

dialog = "extract persona: bot_0: What's up? I just got in from a run..\nbot_1: Not too much just got back from fishing/."
tokens = tokenizer(dialog, return_tensors='pt')

model.eval()
with torch.inference_mode():
    outputs = model.generate(**tokens.to(model.device), max_length=512)
print(tokenizer.batch_decode(outputs))