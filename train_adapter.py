from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import MBart50Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from models_cls.adapter_bart import MBartForConditionalGenerationWithAdapter

# Prepare and tokenize dataset
train = pd.read_json('data/train.jsonl', lines=True)
test = pd.read_json('data/test.jsonl', lines=True)
valid = pd.read_json('data/valid.jsonl', lines=True)

train = Dataset.from_pandas(train).shuffle()
test = Dataset.from_pandas(test).shuffle()
valid = Dataset.from_pandas(valid).shuffle()

# model_name = 'google/flan-t5-small'
model_name = 'facebook/mbart-large-50'

tokenizer = MBart50Tokenizer.from_pretrained(model_name, src_lang='en_XX')
prefix = "{}"

def preprocess_function(examples):
    inputs = [prefix.format(doc) for doc in examples["history"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    labels = tokenizer(examples["persona"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train.map(preprocess_function, batched=True)
tokenized_test = test.map(preprocess_function, batched=True)
tokenized_valid = valid.map(preprocess_function, batched=True)

# Load pretrained model and evaluate model after each epoch
model = MBartForConditionalGenerationWithAdapter.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./models/mbart-large-extract",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    eval_steps=500,
    logging_steps=1,
    gradient_checkpointing=False,
    gradient_accumulation_steps=32,
    lr_scheduler_type='cosine',
    warmup_steps=30,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model('models/mbart-large-extract')