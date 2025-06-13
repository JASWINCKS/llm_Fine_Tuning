import os
import sys
from pathlib import Path
import logging
import subprocess

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command: str, description: str):
    """Run a shell command and log its output."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Command completed successfully: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {description}: {e}")
        logger.error(f"Command output: {e.output}")
        return False

def main():
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Step 1: Collect data
    logger.info("Step 1: Collecting data from various sources...")
    if not run_command(
        "python src/data/collect_data.py",
        "Data collection"
    ):
        logger.error("Data collection failed!")
        return
    
    # Step 2: Preprocess data
    logger.info("Step 2: Preprocessing collected data...")
    if not run_command(
        "python src/data/preprocess.py",
        "Data preprocessing"
    ):
        logger.error("Data preprocessing failed!")
        return
    
    # Step 3: Train model
    logger.info("Step 3: Starting model training...")
    if not run_command(
        "python src/models/train.py",
        "Model training"
    ):
        logger.error("Model training failed!")
        return
    
    # Step 4: Evaluate model
    logger.info("Step 4: Evaluating model...")
    if not run_command(
        "python src/evaluation/evaluate.py",
        "Model evaluation"
    ):
        logger.error("Model evaluation failed!")
        return
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main() 