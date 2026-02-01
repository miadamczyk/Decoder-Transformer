# Decoder-Only Transformer for Shakespeare Language Modeling

This repository provides a PyTorch implementation of a decoder-only Transformer architecture with self-attention, inspired by the original Transformer paper:
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

### Features

- Configurable model depth, embedding size, and number of attention heads
- Support for two multihead attention implementations: custom and PyTorch native
- Optional Weights & Biases (WandB) integration for experiment tracking
- Automatic checkpoint saving based on validation loss
- Fully configurable training via command-line arguments
- Text generation of example Shakespeare-style samples from saved models

This repository is intended for experimentation with decoder-only Transformer on Shakespeare dataset.

# Description of files

### dataset.py

Provides utilities for downloading, processing, and loading the Tiny Shakespeare dataset for character-level language modeling.

Steps:

- Automatically downloads the Tiny Shakespeare dataset if it is not present locally
- Builds a character-level vocabulary from the full corpus and creates string-to-index (stoi) and index-to-string (itos) mappings
- Splits the text corpus into training, validation, and test sets using configurable ratios
- Constructs input–target pairs for autoregressive language modeling
- Provides helper methods for encoding text into token IDs and decoding token IDs back into text
- Implements a custom PyTorch Dataset 

### model.py

Contains the implementation of a decoder-only Transformer architecture

Components:

- `CustomMultiHeadAttention`: Implements scaled dot-product multi-head self-attention from scratch
- `NativeMultiHeadAttention`: Wraps PyTorch’s built-in MultiheadAttention module
- `DecoderBlock`: Standard Transformer decoder block 
- `DecoderOnlyTransformer`:  Full decoder-only Transformer model composed of a stack of decoder blocks

### main.py

Handles the training, validation, and evaluation loop for the decoder-only Transformer model.

Features:

- Parses command-line arguments to configure:
- Implements a training loop, evaluation and testing
- Automatically saves the best model checkpoint based on validation loss
- Reloads the best-performing model at the end of training for final test evaluation

### generate.py

Provides utilities for text generation using a trained decoder-only Transformer model.

Steps:

- Loads a saved model checkpoint and restores model configuration
- Supports autoregressive text generation 
- Allows generation from an initial prompt 
- Generates Shakespeare-style text sequences character by character

### Example Run

To train a decoder-only Transformer on the Shakespeare dataset using `main.py`, you can run the following command:

```bash
python main.py --data_path data/shakespeare.txt --wandb --wandb_project decoder-only-transformer --wandb_run_name run1
 ```

After that, you can generate Shakespeare-like text using `generate.py`. For example, if you have a saved model checkpoint `best_model.pt` in the `models` directory:


```bash
python generate.py --model_dir models --model_name best_model.pt --output_dir generated_text --start_text "ROMEO: " --device cuda
```

## Dependencies and Installation

This project requires Python 3.10+ and the following libraries:

- torch>=2.0.0 — PyTorch deep learning framework  
- tqdm>=4.65.0 — Progress bars for training and evaluation loops  
- wandb>=0.15.0 — Weights & Biases for experiment tracking  
- numpy>=1.25.0 — Numerical computations  

### Installing via pip

Create and activate a virtual environment (optional but recommended):


```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### Installation via Conda

You can create a Conda environment directly from the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate attention
```

## Results

The training and evaluation results for the decoder-only Transformer trained on the Shakespeare dataset with the default training arguments can be viewed on Weights & Biases:

[Transformer Shakespeare runs](https://api.wandb.ai/links/milosz-adamczyk2002/bmei9b75)

The runs include logged training and validation loss curves and final test set performance  

Generated example text using a trained model with custom attention can be found in outputs/text.txt
