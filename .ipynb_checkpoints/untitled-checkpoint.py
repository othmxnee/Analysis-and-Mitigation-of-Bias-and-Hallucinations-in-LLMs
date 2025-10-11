# ============================================
# CELL 1: Title (Change this cell to Markdown - press M)
# ============================================
"""
After creating this notebook, make this cell Markdown (press M) and write:

# My First LLM Project - Getting Started
## Loading and Testing a Language Model

**Date:** October 10, 2025  
**Objective:** Learn Jupyter and load GPT-2 model
"""

# ============================================
# CELL 2: Import Libraries
# ============================================
# This cell imports all the tools we need
# Run this cell first! (Shift + Enter)

print("📦 Importing libraries...")

# Transformers - for loading AI models
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# PyTorch - the deep learning framework
import torch

# For progress bars
from tqdm import tqdm

# For data handling
import pandas as pd
import numpy as np

# For visualization
import matplotlib.pyplot as plt

print("✅ All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")

# ============================================
# CELL 3: Explanation - What is GPT-2?
# ============================================
"""
📚 WHAT IS GPT-2?

GPT-2 is a language model created by OpenAI in 2019.
- It's smaller and simpler than ChatGPT (GPT-3/4)
- Perfect for learning because it's free and runs on normal computers
- It can generate text, complete sentences, and answer questions

Think of it as a "mini ChatGPT" for practice!

SIZE OPTIONS:
- gpt2: 124M parameters (smallest, fastest) ← We'll use this
- gpt2-medium: 355M parameters
- gpt2-large: 774M parameters
- gpt2-xl: 1.5B parameters
"""

# ============================================
# CELL 4: Load the Model
# ============================================
print("🤖 Loading GPT-2 model...")
print("(First time will download ~500MB, be patient!)")

# Step 1: Load the tokenizer
# Tokenizer = converts text into numbers that the model understands
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print("✅ Tokenizer loaded!")

# Step 2: Load the actual model
# This is the AI "brain"
model = GPT2LMHeadModel.from_pretrained('gpt2')
print("✅ Model loaded!")

# Step 3: Set model to evaluation mode
# This means we're using it (not training it)
model.eval()
print("✅ Model ready for use!")

print("\n🎉 Setup complete! Model is ready to generate text!")

# ============================================
# CELL 5: Understanding Tokenization
# ============================================
"""
🔤 WHAT IS TOKENIZATION?

Computers don't understand words - they understand numbers!
Tokenization converts text → numbers

Example:
"Hello world" → [15496, 995]

Let's try it:
"""

# Example text
text = "Hello, my name is Claude"

# Convert to tokens (numbers)
tokens = tokenizer.encode(text)

print(f"Original text: {text}")
print(f"Tokenized: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Convert back to text (to verify it worked)
decoded = tokenizer.decode(tokens)
print(f"Decoded back: {decoded}")

# ============================================
# CELL 6: Your First Text Generation!
# ============================================
"""
🎨 LET'S GENERATE TEXT!

We'll give the model a prompt and let it continue the text.
"""

def generate_text(prompt, max_length=50):
    """
    Generate text from a prompt.
    
    Args:
        prompt (str): The starting text
        max_length (int): How many tokens to generate
    
    Returns:
        str: Generated text
    """
    # Step 1: Convert prompt to tokens
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Step 2: Generate
    # temperature: higher = more creative, lower = more predictable
    # do_sample: True = random, False = always picks most likely word
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Step 3: Convert back to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

# Try it!
prompt = "Artificial intelligence is"
result = generate_text(prompt, max_length=50)

print(f"Prompt: {prompt}")
print(f"\nGenerated text:\n{result}")

# ============================================
# CELL 7: Test Multiple Prompts
# ============================================
"""
🧪 TESTING DIFFERENT PROMPTS

Let's test the model with different types of prompts
to see how it responds.
"""

# Different prompts to test
test_prompts = [
    "The doctor walked into the room. She",
    "The doctor walked into the room. He",
    "The engineer was working on",
    "In the future, technology will",
    "The most important thing about AI is"
]

print("Testing multiple prompts:\n")
print("="*60)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{i}. Prompt: {prompt}")
    result = generate_text(prompt, max_length=40)
    print(f"   Result: {result}")
    print("-"*60)

# ============================================
# CELL 8: Save Results
# ============================================
"""
💾 SAVING RESULTS

Let's save our test results to a file so we can analyze them later.
"""

# Create a list to store results
results = []

for prompt in test_prompts:
    generated = generate_text(prompt, max_length=40)
    results.append({
        'prompt': prompt,
        'generated_text': generated,
        'length': len(generated.split())
    })

# Convert to DataFrame (like an Excel table)
df = pd.DataFrame(results)

# Display the table
print("\nResults Table:")
print(df)

# Save to CSV file
df.to_csv('results/first_test_results.csv', index=False)
print("\n✅ Results saved to 'results/first_test_results.csv'")

# ============================================
# CELL 9: Simple Visualization
# ============================================
"""
📊 VISUALIZING RESULTS

Let's create a simple chart showing the length of generated texts.
"""

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(results)), [r['length'] for r in results])
plt.xlabel('Prompt Number')
plt.ylabel('Number of Words Generated')
plt.title('Length of Generated Texts')
plt.xticks(range(len(results)), [f"Prompt {i+1}" for i in range(len(results))])
plt.tight_layout()

# Save the plot
plt.savefig('results/visualizations/first_generation_lengths.png')
print("✅ Chart saved to 'results/visualizations/first_generation_lengths.png'")

# Show the plot
plt.show()

# ============================================
# CELL 10: Next Steps Summary
# ============================================
"""
🎯 SUMMARY - What You've Learned:

✅ How to use Jupyter Notebook
✅ How to load a language model (GPT-2)
✅ What tokenization is
✅ How to generate text from prompts
✅ How to save results to CSV
✅ How to create simple visualizations

🚀 NEXT STEPS:
1. Test for bias in the model's responses
2. Detect hallucinations (false information)
3. Implement mitigation techniques

💡 EXPERIMENT IDEAS:
- Try different prompts
- Change the temperature parameter (0.1 to 1.5)
- Generate longer texts (increase max_length)
- Test with different topics
"""

print("="*60)
print("🎉 CONGRATULATIONS!")
print("You've successfully loaded and tested your first AI model!")
print("="*60)