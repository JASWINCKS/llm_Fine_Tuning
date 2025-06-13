import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModel
import numpy as np
from typing import List, Dict
import json
from pathlib import Path
import logging
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import nltk
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt_tab')

def load_config(config_path: str) -> Dict:
    """Load evaluation configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(config: Dict):
    """Load the fine-tuned model and tokenizer."""
    base_model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        config['evaluation']['model_path']
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    return model, tokenizer

def calculate_metrics(predictions: List[str], references: List[str]) -> Dict:
    """Calculate evaluation metrics."""
    # Initialize scorers
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    rouge_scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    # Calculate BLEU scores
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        # ROUGE
        scores = rouge_scorer_obj.score(ref, pred)
        for metric in rouge_scores.keys():
            rouge_scores[metric].append(scores[metric].fmeasure)
        
        # BLEU
        reference_tokens = [nltk.word_tokenize(ref)]
        prediction_tokens = nltk.word_tokenize(pred)
        bleu_scores.append(sentence_bleu(reference_tokens, prediction_tokens))
    
    # Calculate averages
    metrics = {
        'bleu': np.mean(bleu_scores),
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL'])
    }
    
    return metrics

def evaluate_model(model, tokenizer, test_data, config: Dict) -> Dict:
    """Evaluate model on test data."""
    predictions = []
    references = []
    
    for example in test_data:
        # Prepare input
        input_text = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=config['model']['max_length'])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config['model']['max_length'],
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = prediction.split("Output:")[-1].strip()
        
        predictions.append(prediction)
        references.append(example['output'])
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, references)
    
    # Save predictions
    results = {
        'metrics': metrics,
        'predictions': [
            {
                'instruction': ex['instruction'],
                'input': ex['input'],
                'reference': ex['output'],
                'prediction': pred
            }
            for ex, pred in zip(test_data, predictions)
        ]
    }
    
    return results

def main():
    # Load configuration
    config = load_config("configs/training_config.yaml")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load test data
    test_data = load_dataset(
        'json',
        data_files=config['evaluation']['test_file'],
        split='train'
    )
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, test_data, config)
    
    # Save results
    output_dir = Path(config['evaluation']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log metrics
    logger.info("Evaluation Metrics:")
    for metric, value in results['metrics'].items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 