import json
import pandas as pd
from typing import List, Dict
import re
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_qa_pair(instruction: str, input_text: str, output: str) -> Dict:
    """Format a Q&A pair into the required structure."""
    return {
        "instruction": clean_text(instruction),
        "input": clean_text(str(input_text) if pd.notna(input_text) else ""),
        "output": clean_text(str(output) if pd.notna(output) else "")
    }

def process_dataset(input_file: str, output_dir: str):
    """Process the dataset and split into train/val/test sets."""
    logger.info(f"Processing dataset from {input_file}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read input data
    # Assuming input is a CSV with columns: instruction, input, output
    df = pd.read_csv(input_file)
    
    # Process each row
    processed_data = []
    for _, row in df.iterrows():
        qa_pair = format_qa_pair(
            row['instruction'],
            row.get('input', ''),
            row['output']
        )
        processed_data.append(qa_pair)
    
    # Split into train/val/test (80/10/10)
    train_data, temp_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Save processed datasets
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = Path(output_dir) / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {split_name} set to {output_file}")

def main():
    # Configuration
    input_file = "data/raw/security_dataset.csv"  # Update with your input file path
    output_dir = "data/processed"
    
    # Process dataset
    process_dataset(input_file, output_dir)
    logger.info("Dataset processing completed!")

if __name__ == "__main__":
    main() 