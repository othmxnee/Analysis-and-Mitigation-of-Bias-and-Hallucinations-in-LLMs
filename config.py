"""
Configuration file for the LLM Bias and Hallucination Analysis project.
This file contains all the important settings in one place.
"""

import os

# ============================================
# PROJECT PATHS
# ============================================
# Get the root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
BIAS_DATA_DIR = os.path.join(DATA_DIR, 'bias_tests')
HALLUCINATION_DATA_DIR = os.path.join(DATA_DIR, 'hallucination_tests')

# Results directories
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
BIAS_RESULTS_DIR = os.path.join(RESULTS_DIR, 'bias_analysis')
HALLUCINATION_RESULTS_DIR = os.path.join(RESULTS_DIR, 'hallucination_analysis')
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'visualizations')

# ============================================
# MODEL SETTINGS
# ============================================
# We'll use GPT-2 as our test model (it's small and free)
MODEL_NAME = "gpt2"  # You can change this later to "gpt2-medium" or other models

# Maximum length of generated text
MAX_LENGTH = 100

# Temperature controls randomness (0.0 = deterministic, 1.0 = very random)
TEMPERATURE = 0.7

# Number of sequences to generate for each prompt
NUM_SEQUENCES = 5

# ============================================
# BIAS TESTING SETTINGS
# ============================================
# Categories of bias to test
BIAS_CATEGORIES = [
    "gender",
    "race",
    "profession",
    "age"
]

# Number of test prompts per category
NUM_BIAS_TESTS_PER_CATEGORY = 10

# ============================================
# HALLUCINATION TESTING SETTINGS
# ============================================
# Types of factual questions to test
HALLUCINATION_TEST_TYPES = [
    "historical_facts",
    "scientific_facts",
    "geographical_facts",
    "mathematical_facts"
]

# Number of questions per type
NUM_HALLUCINATION_TESTS_PER_TYPE = 10

# ============================================
# EVALUATION METRICS
# ============================================
# Metrics we'll calculate
METRICS = [
    "bias_score",
    "hallucination_rate",
    "confidence_accuracy",
    "stereotype_frequency"
]

# ============================================
# VISUALIZATION SETTINGS
# ============================================
# Chart style
PLOT_STYLE = "seaborn"
FIGURE_SIZE = (10, 6)
FONT_SIZE = 12

# Colors for charts
COLOR_PALETTE = [
    "#3498db",  # Blue
    "#e74c3c",  # Red
    "#2ecc71",  # Green
    "#f39c12",  # Orange
    "#9b59b6"   # Purple
]

# ============================================
# HELPER FUNCTIONS
# ============================================

def create_directories():
    """
    Creates all necessary directories if they don't exist.
    Call this at the start of your project.
    """
    directories = [
        DATA_DIR,
        BIAS_DATA_DIR,
        HALLUCINATION_DATA_DIR,
        RESULTS_DIR,
        BIAS_RESULTS_DIR,
        HALLUCINATION_RESULTS_DIR,
        VISUALIZATION_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")

def print_config():
    """
    Prints the current configuration settings.
    Useful for debugging and documentation.
    """
    print("="*60)
    print("PROJECT CONFIGURATION")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Bias Categories: {', '.join(BIAS_CATEGORIES)}")
    print(f"Hallucination Test Types: {', '.join(HALLUCINATION_TEST_TYPES)}")
    print("="*60)

# Run this when the config file is executed directly
if __name__ == "__main__":
    print("Setting up project directories...")
    create_directories()
    print("\nCurrent configuration:")
    print_config()
    print("\n✓ Configuration complete!")