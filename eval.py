import argparse
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from tqdm import tqdm

from models import DKT, DKTConfig, SAKT, SAKTConfig
from train import XES3G5MDataModule, XES3G5MDataModuleConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Knowledge Tracing models")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, choices=["dkt", "sakt"], default="dkt", help="Model type")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers (for DKT)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (for SAKT)")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--val_fold", type=int, default=4, help="Validation fold")
    parser.add_argument("--max_seq_length", type=int, default=200, help="Maximum sequence length")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./evaluation", help="Output directory for evaluation results")
    parser.add_argument("--dataset_split", type=str, choices=["test", "val"], default="test", help="Dataset split to evaluate on")
    
    return parser.parse_args()


def get_predictions(model, dataloader):
    """
    Get model predictions on a dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader with test data
        
    Returns:
        tuple: (y_true, y_pred) - ground truth and predicted values
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_questions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            questions = batch['questions']
            responses = batch['responses']
            selectmasks = batch['selectmasks']
            
            # Forward pass
            pred = model(questions, responses, selectmasks)
            
            # Get targets for all questions after the first step
            target_questions = questions[:, 1:]
            target_responses = responses[:, 1:].float()
            target_selectmasks = selectmasks[:, 1:]
            
            # For DKT, we need to gather predictions for the target questions
            if isinstance(model, DKT):
                batch_size, seq_len = target_questions.size()
                pred_flat = pred.view(batch_size * seq_len, -1)
                target_questions_flat = target_questions.view(-1)
                pred_selected = pred_flat[torch.arange(batch_size * seq_len), target_questions_flat]
                pred_selected = pred_selected.view(batch_size, seq_len)
                pred = pred_selected
            
            # Apply mask
            valid_mask = (target_selectmasks == 1) & (target_questions != -1)
            
            if valid_mask.sum() > 0:
                # Flatten the tensors
                pred_flat = pred[valid_mask].cpu().numpy()
                target_flat = target_responses[valid_mask].cpu().numpy()
                questions_flat = target_questions[valid_mask].cpu().numpy()
                
                all_preds.append(pred_flat)
                all_targets.append(target_flat)
                all_questions.append(questions_flat)
    
    # Concatenate all predictions and targets
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    questions = np.concatenate(all_questions)
    
    return y_true, y_pred, questions


def evaluate_model(model, dataloader):
    """
    Evaluate model performance on a dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader with test data
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    y_true, y_pred, questions = get_predictions(model, dataloader)
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    
    # Calculate AUPR (area under precision-recall curve)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    
    # Calculate accuracy
    accuracy = np.mean((y_pred >= 0.5) == y_true)
    
    return {
        'auc': auc_score,
        'aupr': aupr,
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'y_true': y_true,
        'y_pred': y_pred,
        'questions': questions
    }


def plot_roc_curve(eval_results, output_dir):
    """
    Plot ROC curve from evaluation results.
    
    Args:
        eval_results: Dictionary with evaluation metrics
        output_dir: Output directory for plots
    """
    plt.figure(figsize=(8, 8))
    plt.plot(eval_results['fpr'], eval_results['tpr'], lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {eval_results["auc"]:.4f})')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curve(eval_results, output_dir):
    """
    Plot precision-recall curve from evaluation results.
    
    Args:
        eval_results: Dictionary with evaluation metrics
        output_dir: Output directory for plots
    """
    plt.figure(figsize=(8, 8))
    plt.plot(eval_results['recall'], eval_results['precision'], lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUPR = {eval_results["aupr"]:.4f})')
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_distribution(eval_results, output_dir):
    """
    Plot histogram of predicted probabilities.
    
    Args:
        eval_results: Dictionary with evaluation metrics
        output_dir: Output directory for plots
    """
    plt.figure(figsize=(10, 6))
    
    # Separate predictions for correct and incorrect answers
    pred_correct = eval_results['y_pred'][eval_results['y_true'] == 1]
    pred_incorrect = eval_results['y_pred'][eval_results['y_true'] == 0]
    
    # Plot histograms
    plt.hist(pred_correct, bins=50, alpha=0.5, label='Correct answers')
    plt.hist(pred_incorrect, bins=50, alpha=0.5, label='Incorrect answers')
    
    plt.xlabel('Predicted probability')
    plt.ylabel('Count')
    plt.title('Distribution of predicted probabilities')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_by_question_frequency(eval_results, dataloader, output_dir):
    """
    Analyze performance by question frequency.
    
    Args:
        eval_results: Dictionary with evaluation metrics
        dataloader: DataLoader with test data
        output_dir: Output directory for plots
    """
    # Count question frequencies in the training data
    question_count = {}
    for batch in dataloader.train_dataloader():
        questions = batch['questions'].cpu().numpy()
        for q in questions.flatten():
            if q != -1:  # Skip padding
                question_count[q] = question_count.get(q, 0) + 1
    
    # Get questions from evaluation results
    eval_questions = eval_results['questions']
    y_true = eval_results['y_true']
    y_pred = eval_results['y_pred']
    
    # Group questions by frequency
    frequency_bins = [0, 10, 50, 100, 500, np.inf]
    bin_names = ['1-10', '11-50', '51-100', '101-500', '500+']
    
    metrics_by_freq = {name: {'auc': [], 'accuracy': [], 'count': 0} for name in bin_names}
    
    for q, t, p in zip(eval_questions, y_true, y_pred):
        freq = question_count.get(q, 0)
        
        for i, (low, high) in enumerate(zip(frequency_bins[:-1], frequency_bins[1:])):
            if low < freq <= high:
                bin_name = bin_names[i]
                metrics_by_freq[bin_name]['count'] += 1
                
                # Collect predictions for this frequency bin
                metrics_by_freq[bin_name]['true'] = metrics_by_freq[bin_name].get('true', []) + [t]
                metrics_by_freq[bin_name]['pred'] = metrics_by_freq[bin_name].get('pred', []) + [p]
                break
    
    # Calculate metrics for each frequency bin
    results = []
    for bin_name, data in metrics_by_freq.items():
        if len(data.get('true', [])) > 10:  # Only calculate metrics if we have enough samples
            data_true = np.array(data.get('true', []))
            data_pred = np.array(data.get('pred', []))
            
            try:
                # Check if we have at least two different classes to compute AUC
                if len(np.unique(data_true)) >= 2:
                    bin_auc = roc_auc_score(data_true, data_pred)
                else:
                    bin_auc = float('nan')
                    
                bin_accuracy = np.mean((data_pred >= 0.5) == data_true)
            except Exception as e:
                print(f"Error calculating metrics for bin {bin_name}: {e}")
                bin_auc = float('nan')
                bin_accuracy = float('nan')
                
            results.append({
                'frequency_bin': bin_name,
                'count': data['count'],
                'auc': bin_auc,
                'accuracy': bin_accuracy
            })
    
    # Create a DataFrame and plot
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='frequency_bin', y='auc', data=df)
    plt.title('AUC by Question Frequency')
    plt.ylabel('AUC')
    plt.xlabel('Question Frequency')
    plt.savefig(os.path.join(output_dir, 'auc_by_question_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='frequency_bin', y='accuracy', data=df)
    plt.title('Accuracy by Question Frequency')
    plt.ylabel('Accuracy')
    plt.xlabel('Question Frequency')
    plt.savefig(os.path.join(output_dir, 'accuracy_by_question_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    df.to_csv(os.path.join(output_dir, 'metrics_by_question_frequency.csv'), index=False)


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up data module
    data_config = XES3G5MDataModuleConfig()
    data_config.batch_size = args.batch_size
    data_config.val_fold = args.val_fold
    data_config.max_seq_length = args.max_seq_length
    
    data_module = XES3G5MDataModule(config=data_config)
    data_module.prepare_data()
    data_module.setup()
    
    # Get number of questions and concepts
    question_content_df = pd.DataFrame(data_module.datasets["content_metadata"]["question"])
    concept_content_df = pd.DataFrame(data_module.datasets["content_metadata"]["concept"])
    num_questions = len(question_content_df)
    num_concepts = len(concept_content_df)
    
    # Load model
    if args.model_type == "dkt":
        model_config = DKTConfig()
        model_config.hidden_dim = args.hidden_dim
        model_config.num_layers = args.num_layers
        
        model = DKT(
            num_questions=num_questions,
            num_concepts=num_concepts,
            config=model_config
        )
    else:
        model_config = SAKTConfig()
        model_config.hidden_dim = args.hidden_dim
        model_config.num_heads = args.num_heads
        
        model = SAKT(
            num_questions=num_questions,
            num_concepts=num_concepts,
            config=model_config
        )
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path))
    
    # Select the right dataloader based on the dataset split
    if args.dataset_split == "test":
        dataloader = data_module.test_dataloader()
    else:
        dataloader = data_module.val_dataloader()
    
    # Evaluate model
    eval_results = evaluate_model(model, dataloader)
    
    # Print results
    print(f"Model: {args.model_type}")
    print(f"Dataset: {args.dataset_split}")
    print(f"AUC: {eval_results['auc']:.4f}")
    print(f"AUPR: {eval_results['aupr']:.4f}")
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    
    # Save results to a file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Dataset: {args.dataset_split}\n")
        f.write(f"AUC: {eval_results['auc']:.4f}\n")
        f.write(f"AUPR: {eval_results['aupr']:.4f}\n")
        f.write(f"Accuracy: {eval_results['accuracy']:.4f}\n")
    
    # Generate plots
    plot_roc_curve(eval_results, args.output_dir)
    plot_pr_curve(eval_results, args.output_dir)
    plot_prediction_distribution(eval_results, args.output_dir)
    
    # Additional analysis
    try:
        analyze_by_question_frequency(eval_results, data_module, args.output_dir)
    except Exception as e:
        print(f"Error in question frequency analysis: {e}")
    
    print(f"Evaluation results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 