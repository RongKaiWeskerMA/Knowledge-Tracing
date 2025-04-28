import argparse
import os
import numpy as np
import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from datasets import load_dataset
from models import DKT, DKTConfig, SAKT, SAKTConfig


class XES3G5MDataModuleConfig:
    """Configuration for the data module."""

    hf_dataset_ids: dict[str, str] = {
        "sequence": "Atomi/XES3G5M_interaction_sequences",
        "content_metadata": "Atomi/XES3G5M_content_metadata",
    }
    max_seq_length: int = 200
    padding_value: int = -1
    batch_size: int = 64
    val_fold: int = 4


class XES3G5MDataset(torch.utils.data.Dataset):
    """
    Dataset class for XES3G5M dataset.
    """

    def __init__(self, seq_df, question_embeddings, concept_embeddings):
        """
        Initializes the dataset.
        """
        self.seq_df = seq_df
        self.question_embeddings = question_embeddings
        self.concept_embeddings = concept_embeddings

    def __len__(self) -> int:
        return len(self.seq_df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a dictionary of input tensors.
        """
        row = self.seq_df.iloc[idx]
        question_embeddings = self.question_embeddings[row["questions"]]  # (padded_num_questions, emb_dim)
        concept_embeddings = self.concept_embeddings[row["concepts"]]  # (padded_num_concepts, emb_dim)
        selectmasks = row["selectmasks"]
        responses = row["responses"]
        return {
            "questions": torch.LongTensor(row["questions"]),
            "concepts": torch.LongTensor(row["concepts"]),
            "question_embeddings": torch.Tensor(question_embeddings),
            "concept_embeddings": torch.Tensor(concept_embeddings),
            "selectmasks": torch.Tensor(selectmasks),
            "responses": torch.LongTensor(responses),
        }


class XES3G5MDataModule(pl.LightningDataModule):
    """
    DataModule class for XES3G5M dataset.
    """

    def __init__(
        self,
        config: XES3G5MDataModuleConfig,
    ) -> None:
        """
        Initializes the data module.
        """
        super().__init__()
        self.hf_dataset_ids = config.hf_dataset_ids
        self.batch_size = config.batch_size
        self.val_fold = config.val_fold
        self.max_seq_length = config.max_seq_length
        self.padding_value = config.padding_value

    def prepare_data(self) -> None:
        """
        Downloads the dataset.
        """
        [load_dataset(hf_dataset_id) for hf_dataset_id in self.hf_dataset_ids.values()]

    def setup(self, stage: str | None = None) -> None:
        """
        Loads the dataset.
        """
        datasets = {key: load_dataset(value) for key, value in self.hf_dataset_ids.items()}
        self.datasets = datasets

        seq_df_train_val = datasets["sequence"]["train"].to_pandas()
        val_indices = seq_df_train_val["fold"] == self.val_fold
        self.seq_df_val = seq_df_train_val[val_indices]
        self.seq_df_train = seq_df_train_val[~val_indices]
        self.seq_df_test = datasets["sequence"]["test"].to_pandas()

        question_content_df = datasets["content_metadata"]["question"].to_pandas()
        concept_content_df = datasets["content_metadata"]["concept"].to_pandas()
        self.question_embeddings = np.array([np.array(x) for x in question_content_df["embeddings"].values])
        self.concept_embeddings = np.array([np.array(x) for x in concept_content_df["embeddings"].values])

        if stage == "fit" or stage is None:
            self.train_dataset = XES3G5MDataset(
                seq_df=self.seq_df_train,
                question_embeddings=self.question_embeddings,
                concept_embeddings=self.concept_embeddings,
            )
            self.val_dataset = XES3G5MDataset(
                seq_df=self.seq_df_val,
                question_embeddings=self.question_embeddings,
                concept_embeddings=self.concept_embeddings,
            )
        if stage == "test" or stage is None:
            self.test_dataset = XES3G5MDataset(
                seq_df=self.seq_df_test,
                question_embeddings=self.question_embeddings,
                concept_embeddings=self.concept_embeddings,
            )

    def _collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Collate function for the dataloader.
        """

        # Get list of tensors from the batch
        questions = [x["questions"] for x in batch]
        concepts = [x["concepts"] for x in batch]
        question_embeddings = [x["question_embeddings"] for x in batch]
        concept_embeddings = [x["concept_embeddings"] for x in batch]
        selectmasks = [x["selectmasks"] for x in batch]
        responses = [x["responses"] for x in batch]

        # Get the maximum sequence length in this batch
        max_len = max(x.shape[0] for x in questions)
        max_len = min(max_len, self.max_seq_length)  # Cap at max_seq_length if needed

        # Pad the sequences if not done already (for "test" mode)
        for i in range(len(questions)):
            seq_len = questions[i].shape[0]
            if seq_len < max_len:
                questions[i] = torch.nn.functional.pad(questions[i], (0, max_len - seq_len), value=self.padding_value)
                concepts[i] = torch.nn.functional.pad(concepts[i], (0, max_len - seq_len), value=self.padding_value)
                question_embeddings[i] = torch.nn.functional.pad(
                    question_embeddings[i], (0, 0, 0, max_len - seq_len), value=0
                )
                concept_embeddings[i] = torch.nn.functional.pad(
                    concept_embeddings[i], (0, 0, 0, max_len - seq_len), value=0
                )
                selectmasks[i] = torch.nn.functional.pad(selectmasks[i], (0, max_len - seq_len), value=0)
                responses[i] = torch.nn.functional.pad(responses[i], (0, max_len - seq_len), value=self.padding_value)
            else:
                questions[i] = questions[i][:max_len]
                concepts[i] = concepts[i][:max_len]
                question_embeddings[i] = question_embeddings[i][:max_len]
                concept_embeddings[i] = concept_embeddings[i][:max_len]
                selectmasks[i] = selectmasks[i][:max_len]
                responses[i] = responses[i][:max_len]

        # Stack the tensors
        stacked_questions = torch.stack(questions)  # (batch_size, max_seq_length)
        stacked_concepts = torch.stack(concepts)  # (batch_size, max_seq_length)
        stacked_question_embeddings = torch.stack(question_embeddings)  # (batch_size, max_seq_length, emb_dim)
        stacked_concept_embeddings = torch.stack(concept_embeddings)  # (batch_size, max_seq_length, emb_dim)
        stacked_selectmasks = torch.stack(selectmasks)  # (batch_size, max_seq_length)
        stacked_responses = torch.stack(responses)  # (batch_size, max_seq_length)

        # Replace padding value with 0 for responses
        stacked_responses[stacked_responses == self.padding_value] = 0
        
        # Set any negative question IDs to 0 to prevent out-of-bounds indexing
        stacked_questions[stacked_questions < 0] = 0

        return {
            "questions": stacked_questions,
            "concepts": stacked_concepts,
            "question_embeddings": stacked_question_embeddings,
            "concept_embeddings": stacked_concept_embeddings,
            "selectmasks": stacked_selectmasks,
            "responses": stacked_responses,
        }

    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=8,
        )

    def val_dataloader(self):
        """
        Returns the validation dataloader.
        """
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=8,
        )

    def test_dataloader(self):
        """
        Returns the test dataloader.
        """
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=8,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Knowledge Tracing models")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, choices=["dkt", "sakt"], default="sakt", help="Model type")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers (for DKT)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (for SAKT)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--use_pretrained_embeddings", type=bool, default=True, help="Use pretrained embeddings")
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--val_fold", type=int, default=4, help="Validation fold")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    
    # Data parameters
    parser.add_argument("--max_seq_length", type=int, default=200, help="Maximum sequence length")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    
    # CUDA settings
    parser.add_argument("--precision", type=str, default="32-true", 
                        choices=["32-true", "16-mixed", "bf16-mixed"], 
                        help="Precision for training (16-mixed or bf16-mixed recommended for GPU)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up output directory
    if args.experiment_name is None:
        args.experiment_name = f"{args.model_type}_h{args.hidden_dim}_bs{args.batch_size}_lr{args.learning_rate}"
    
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Print training configuration
    print("\n=== Training Configuration ===")
    print(f"Model Type: {args.model_type}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Precision: {args.precision}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"Output Directory: {output_dir}")
    print("============================\n")
    
    # Set up data module
    data_config = XES3G5MDataModuleConfig()
    data_config.batch_size = args.batch_size
    data_config.val_fold = args.val_fold
    data_config.max_seq_length = args.max_seq_length
    
    data_module = XES3G5MDataModule(config=data_config)
    
    # Temporarily set up data module to get dataset info
    data_module.prepare_data()
    data_module.setup()
    
    # Get number of questions and concepts
    question_content_df = load_dataset(data_config.hf_dataset_ids["content_metadata"])["question"].to_pandas()
    concept_content_df = load_dataset(data_config.hf_dataset_ids["content_metadata"])["concept"].to_pandas()
    num_questions = len(question_content_df)
    num_concepts = len(concept_content_df)
    
    print(f"Dataset loaded: {num_questions} questions, {num_concepts} concepts")
    
    # Set up model
    if args.model_type == "dkt":
        model_config = DKTConfig()
        model_config.hidden_dim = args.hidden_dim
        model_config.num_layers = args.num_layers
        model_config.seq_length = args.max_seq_length
        model_config.dropout = args.dropout
        model_config.learning_rate = args.learning_rate
        model_config.weight_decay = args.weight_decay
        model_config.use_pretrained_embeddings = args.use_pretrained_embeddings
        
        model = DKT(
            num_questions=num_questions,
            num_concepts=num_concepts,
            config=model_config
        )
    else:
        model_config = SAKTConfig()
        model_config.hidden_dim = args.hidden_dim
        model_config.num_heads = args.num_heads
        model_config.dropout = args.dropout
        model_config.learning_rate = args.learning_rate
        model_config.weight_decay = args.weight_decay
        
        model = SAKT(
            num_questions=num_questions,
            num_concepts=num_concepts,
            config=model_config
        )
    
    # Set up logger and callbacks
    logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=3
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_auc",
        patience=args.patience,
        mode="max",
        verbose=True
    )
    
    # Set up trainer with reduced memory usage settings
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        precision=args.precision,        # Use mixed precision to reduce memory
        gradient_clip_val=1.0,           # Gradient clipping to prevent exploding gradients
        accumulate_grad_batches=2,       # Accumulate gradients to effectively reduce batch size
        deterministic=False,             # Disable deterministic mode for better performance
    )
    
    # Print model summary before training
    try:
        print("Model Summary:")
        summary_data = str(model)
        summary_data = summary_data.split("\n")
        for line in summary_data[:10]:  # Only print first 10 lines
            print(line)
        print("...")
    except Exception as e:
        print(f"Could not print model summary: {e}")
    
    # Train model with error handling
    try:
        print(f"\nStarting training with {args.precision} precision...")
        trainer.fit(model, data_module)
        
        # Test model
        print("\nTraining complete, evaluating model on test set...")
        trainer.test(datamodule=data_module, ckpt_path="best")
        
        # Save final model and config
        final_model_path = os.path.join(output_dir, "final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        
        # Print final results
        print(f"\nTraining Results:")
        print(f"Best model saved to {checkpoint_callback.best_model_path}")
        print(f"Best validation AUC: {checkpoint_callback.best_model_score:.4f}")
        print(f"Final model saved to {final_model_path}")
        
        # Return best validation score
        return checkpoint_callback.best_model_score
    
    except RuntimeError as e:
        print(f"RuntimeError during training: {e}")
        if "CUDA" in str(e):
            print("\nTips to resolve CUDA errors:")
            print("1. Try reducing batch size (--batch_size)")
            print("2. Try reducing model size (--hidden_dim)")
            print("3. Use mixed precision training (--precision 16-mixed)")
            print("4. Try using CPU training by disabling CUDA: CUDA_VISIBLE_DEVICES=''")
            print("5. Ensure your GPU has enough memory for this task")
        return None


if __name__ == "__main__":
    main() 