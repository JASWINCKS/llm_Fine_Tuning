import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from typing import Dict, Any, Tuple, List
import yaml
import json

# Check GPU availability
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())

def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_model_and_tokenizer(config: Dict[str, Any]):
    """Initialize model and tokenizer with specified configuration."""
    # Load base model and tokenizer
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Print device placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config['training']['use_fp16'] else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Set model's padding token to match tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id

    # Configure LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def prepare_dataset(config: Dict[str, Any], tokenizer) -> Tuple[Dataset, Dataset]:
    """Load and prepare train and test datasets.
    
    Args:
        config: Configuration dictionary containing data paths and settings
        tokenizer: Tokenizer to use for processing the text
        
    Returns:
        Tuple of (train_dataset, test_dataset) containing processed and tokenized data
    """
    def load_json_dataset(path: str) -> Dataset:
        """Load a JSON file into a HuggingFace Dataset."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_list(data)

    def format_example(example: Dict[str, str]) -> str:
        """Format a single example into a text string.
        
        Args:
            example: Dictionary containing instruction, input, and output
            
        Returns:
            Formatted text string
        """
        parts = [f"Instruction: {example['instruction']}"]
        
        if example.get('input'):
            parts.append(f"Input: {example['input']}")
            
        if example.get('output'):
            parts.append(f"Output: {example['output']}")
            
        return "\n".join(parts)

    def process_batch(examples: Dict[str, List]) -> Dict[str, List[int]]:
        """Process a batch of examples.
        
        Args:
            examples: Dictionary containing lists of values for each field
            
        Returns:
            Dictionary containing tokenized inputs
        """
        # Convert batch dictionary to list of examples
        batch_size = len(examples['instruction'])
        formatted_examples = []
        
        for i in range(batch_size):
            example = {
                'instruction': examples['instruction'][i],
                'input': examples['input'][i] if 'input' in examples else '',
                'output': examples['output'][i] if 'output' in examples else ''
            }
            formatted_examples.append(format_example(example))
        
        # Tokenize the formatted texts
        return tokenizer(
            formatted_examples,
            padding="max_length",
            truncation=True,
            max_length=config['data']['max_length'],
            return_tensors=None  # Return lists instead of tensors
        )

    # Load the datasets
    print("Loading datasets...")
    train_dataset = load_json_dataset(config['data']['train_path'])
    test_dataset = load_json_dataset(config['data']['test_path'])
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Process the datasets
    print("Processing datasets...")
    train_dataset = train_dataset.map(
        process_batch,
        batched=True,
        batch_size=100,  # Process in batches of 100
        remove_columns=train_dataset.column_names,
        desc="Processing train dataset"
    )
    
    test_dataset = test_dataset.map(
        process_batch,
        batched=True,
        batch_size=100,  # Process in batches of 100
        remove_columns=test_dataset.column_names,
        desc="Processing test dataset"
    )

    print("Dataset preparation complete!")
    return train_dataset, test_dataset

def main():
    # Load configuration
    config = load_config("configs/training_config.yaml")

    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(config)

    # Prepare datasets
    train_dataset, test_dataset = prepare_dataset(config, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        # Output and save settings
        output_dir=config['training']['output_dir'],
        save_steps=config['training']['save_steps'],
        
        # Training settings
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        warmup_steps=config['training']['warmup_steps'],
        
        # Optimization settings
        fp16=config['training']['use_fp16'] and torch.cuda.is_available(),  # Only use fp16 if GPU available
        
        # Logging settings
        logging_steps=config['training']['logging_steps'],
        logging_dir=os.path.join(config['training']['output_dir'], "logs"),
        report_to="none",
        
        # Other settings
        seed=config['training']['seed'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=torch.cuda.is_available(),  # Only use pin_memory if GPU available
        no_cuda=not torch.cuda.is_available()  # Explicitly set no_cuda based on GPU availability
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Start training
    trainer.train()

    # Save final model
    trainer.save_model(config['training']['output_dir'])

if __name__ == "__main__":
    main() 