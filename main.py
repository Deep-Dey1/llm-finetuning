import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import datasets
import wandb
import evaluate
import numpy as np

# Initialize WandB
wandb.init(project="llm_finetuning", name="Mistral7B_Finetuning")

# Load model and tokenizer (Mistral 7B or any selected model)
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model with 4-bit quantization for RTX 3080 Ti
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,  # Using QLoRA for efficiency
    torch_dtype=torch.float16,
    device_map="auto"
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Apply LoRA Configuration
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load dataset
DATASET_NAME = "stackexchange/cs"  # Specify your dataset
dataset = datasets.load_dataset(DATASET_NAME)
train_test_split = dataset["data"].train_test_split(test_size=0.2)
train_data = train_test_split["train"]
val_data = train_test_split["test"]

def tokenize_function(examples):
    return tokenizer(examples["question"] + " " + examples["answer"], padding="max_length", truncation=True)

train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)

# Training arguments optimized for RTX 3080 Ti
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=1,  # Reduce batch size for memory efficiency
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    report_to="wandb",
    save_total_limit=2,
    load_best_model_at_end=True,  # For early stopping
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    gradient_checkpointing=True  # Reduce memory usage
)

# Define Trainer
def compute_metrics(eval_pred):
    logits, labels, _ = eval_pred  # Correct unpacking
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

# Evaluate the model
outputs = trainer.predict(val_data)
logits, labels = outputs.predictions, outputs.label_ids
predictions = np.argmax(logits, axis=-1)

precision = evaluate.load("precision").compute(predictions=predictions, references=labels)
recall = evaluate.load("recall").compute(predictions=predictions, references=labels)
f1 = evaluate.load("f1").compute(predictions=predictions, references=labels)

wandb.log({"precision": precision, "recall": recall, "f1_score": f1})

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(labels, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

print("Training and evaluation complete!")
