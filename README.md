# Knowledge Tracing with Deep Learning

A PyTorch implementation of deep learning models for knowledge tracing to predict student performance.

## Overview

Knowledge Tracing (KT) is a fundamental technology in intelligent tutoring systems used to:
- Simulate changes in students' knowledge state during learning
- Track personalized knowledge mastery 
- Predict student performance on future questions

This project implements and compares two state-of-the-art deep learning approaches for knowledge tracing:
1. **Deep Knowledge Tracing (DKT)**: Using recurrent neural networks (LSTM)
2. **Self-Attentive Knowledge Tracing (SAKT)**: Using transformer-based attention mechanisms

## Dataset

The models are trained and evaluated on the **XES3G5M dataset**, a large-scale knowledge tracing benchmark:

- **7,652** unique questions
- **865** knowledge concepts
- **5,549,635** learning interactions
- **18,066** students

The dataset is loaded using Hugging Face datasets:
- `Atomi/XES3G5M_interaction_sequences`: Contains student interaction sequences
- `Atomi/XES3G5M_content_metadata`: Contains metadata and embeddings for questions and concepts

## Model Architectures

### Deep Knowledge Tracing (DKT)

DKT uses recurrent neural networks to model the sequential nature of student learning:

- LSTM-based architecture to capture temporal dependencies
- Question embedding layer that combines question ID with correctness
- Configurable hidden dimension and number of layers
- Binary output prediction for each question

### Self-Attentive Knowledge Tracing (SAKT)

SAKT leverages the transformer architecture and attention mechanisms:

- Multi-head self-attention to capture relationships between exercises
- Position embeddings to maintain sequential information
- Exercise embedding layer that captures both question ID and response
- Separate embeddings for exercises to predict
- Feed-forward network for final prediction

## Training Process

Both models are implemented using PyTorch Lightning for efficient training:

- **Loss function**: Binary Cross-Entropy with Logits
- **Optimizer**: Adam with learning rate scheduling
- **Monitoring**: Area Under ROC Curve (AUC) for validation performance
- **Early stopping**: Based on validation AUC
- **Mixed precision training**: Support for 16-bit and BFloat16 precision
- **Gradient clipping**: To prevent exploding gradients
- **Flexible batch size**: With gradient accumulation

## Evaluation Metrics

Models are evaluated using several metrics:

- **AUC** (Area Under ROC Curve): Primary metric for model performance
- **Accuracy**: Percentage of correctly predicted responses
- **AUPR** (Area Under Precision-Recall Curve): For imbalanced class distributions
- **Loss**: Binary cross-entropy loss

Additionally, the evaluation includes analysis by question frequency to examine model performance on rare vs. common questions.

## Results

Evaluation results include:

- Performance metrics on validation and test sets
- ROC and precision-recall curves
- Visualization of prediction distributions
- Analysis of performance by question frequency

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

To train a model, use the `train.py` script:

```bash
# Train DKT model
python train.py --model_type dkt --hidden_dim 256 --batch_size 64 --learning_rate 1e-3

# Train SAKT model with mixed precision
python train.py --model_type sakt --hidden_dim 256 --num_heads 8 --precision 16-mixed
```

Key parameters:
- `--model_type`: Model architecture (`dkt` or `sakt`)
- `--hidden_dim`: Hidden dimension size
- `--num_layers`: Number of LSTM layers (for DKT)
- `--num_heads`: Number of attention heads (for SAKT)
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--precision`: Training precision (`32-true`, `16-mixed`, or `bf16-mixed`)
- `--output_dir`: Output directory
- `--max_epochs`: Maximum training epochs

### Evaluation

To evaluate a trained model, use the `eval.py` script:

```bash
# Evaluate DKT model
python eval.py --model_type dkt --model_path ./output/dkt_h256_bs64_lr0.001/checkpoints/best.ckpt --output_dir ./evaluation_results

# Evaluate SAKT model on validation set
python eval.py --model_type sakt --model_path ./output/sakt_h256_bs64_lr0.001/checkpoints/best.ckpt --dataset_split val
```

Key parameters:
- `--model_type`: Model type to evaluate
- `--model_path`: Path to the trained model checkpoint
- `--dataset_split`: Dataset split to evaluate on (`test` or `val`)
- `--output_dir`: Directory for evaluation results

## Troubleshooting

If you encounter CUDA memory issues:
1. Reduce batch size (`--batch_size`)
2. Reduce model complexity (`--hidden_dim`)
3. Use mixed precision training (`--precision 16-mixed`)
4. Run on CPU if GPU memory is insufficient

## Implementation Details

- **PyTorch Lightning**: For training organization and acceleration
- **Hugging Face Datasets**: For efficient data loading
- **Mixed precision training**: For faster training and reduced memory usage
- **Robust error handling**: To prevent training failures


## To Do
The implementation of the self-attention knowledge tracing has been completed yet.  

## References

1. [Deep Knowledge Tracing](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)
2. [A Self-Attentive model for Knowledge Tracing](https://arxiv.org/pdf/1907.06837)
3. [XES3G5M Dataset](https://proceedings.neurips.cc/paper_files/paper/2023/file/67fc628f17c2ad53621fb961c6bafcaf-Paper-Datasets_and_Benchmarks.pdf) 