from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load a sample dataset
dataset = load_dataset('glue', 'sst2')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
)

# Trainer
trainer = Trainer(
    model=model,                        
    args=training_args,                  
    train_dataset=dataset['train'],     
    eval_dataset=dataset['validation']   
)

# Start training
trainer.train()
