import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.nn.functional import one_hot

class DKTConfig:
    """Configuration for the DKT model."""
    
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    use_pretrained_embeddings: bool = False

class DKT(pl.LightningModule):
    """Deep Knowledge Tracing model implementation."""
    
    def __init__(
        self, 
        num_questions: int,
        num_concepts: int,
        config: DKTConfig = DKTConfig(),
    ):
        """
        Initialize the DKT model.
        
        Args:
            num_questions: Number of unique questions in the dataset
            num_concepts: Number of unique knowledge concepts in the dataset
            config: Configuration for the model
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.use_pretrained_embeddings = config.use_pretrained_embeddings

        # Input embedding layer (question_id * 2 + correctness)
        # For each question, we create two embeddings: one for correct and one for incorrect
        self.question_map = nn.Linear(2, self.hidden_dim)

        # self.response_embedding = nn.Embedding(
        #     num_embeddings=2,
        #     embedding_dim=self.hidden_dim
        # )   
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Output layer
        self.out = nn.Linear(self.hidden_dim, num_questions)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, questions, responses, selectmasks=None):
        """
        Forward pass through the model.
        
        Args:
            questions: Tensor of question IDs [batch_size, seq_len]
            responses: Tensor of response correctness [batch_size, seq_len]
            selectmasks: Tensor indicating which positions to select for prediction [batch_size, seq_len]
            
        Returns:
            pred: Tensor of logits [batch_size, seq_len, num_questions]
        """
        
           
        

        if self.use_pretrained_embeddings:
            batch_size, seq_len, embedding_dim = questions.size()
            correct_one_hot = one_hot(responses, num_classes=2).float()
            response_encoded = self.question_map(correct_one_hot)
            question_with_correctness = questions + response_encoded
            embedded = question_with_correctness[:, :-1, :]

        else:
            # Create input embeddings based on question + correctness
            # question_ids * 2 is for incorrect, question_ids * 2 + 1 is for correct
            batch_size, seq_len = questions.size()
            question_with_correctness = questions * 2 + responses
        
            # For the input sequence, we use all except the last item
            inp = question_with_correctness[:, :-1, :]
            embedded = self.question_embedding(inp)
            # Handle edge cases where sequence length is 1
            if inp.size(1) == 0:
                # Return zero predictions if input is empty
                return torch.zeros(batch_size, 0, self.num_questions, device=questions.device)
            
        # Apply dropout
        embedded = self.dropout_layer(embedded)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # Get predictions for each question
        pred = self.out(lstm_out)
        
        # Return logits (no sigmoid - will be applied in loss function)
        return pred
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Batch of data from the data loader
            batch_idx: Index of the batch
            
        Returns:
            loss: Training loss
        """
         
        questions = batch['questions']
        responses = batch['responses']
        selectmasks = batch['selectmasks']
        questions_embeddings = batch['question_embeddings']
        
      
        
        # Skip batches with too short sequences
        if questions.size(1) <= 1:
            return None
            
        # Get model predictions
        if self.use_pretrained_embeddings:
            pred = self(questions_embeddings, responses, selectmasks)
        else:
            pred = self(questions, responses, selectmasks)
        
        # Calculate targets for all questions after the first step
        target_questions = questions[:, 1:]
        target_responses = responses[:, 1:].float()
        target_selectmasks = selectmasks[:, 1:]
        
        # Gather predictions for the target questions
        batch_size, seq_len = target_questions.size()
        # Ensure tensor is contiguous before reshaping
        pred_flat = pred.contiguous().reshape(batch_size * seq_len, -1)
        target_questions_flat = target_questions.contiguous().reshape(-1)
        
        # Handle out-of-bounds indices
        valid_indices = (target_questions_flat >= 0) & (target_questions_flat < self.num_questions)
        if not valid_indices.all():
            target_questions_flat = torch.clamp(target_questions_flat, 0, self.num_questions - 1)
            
        pred_selected = pred_flat[torch.arange(batch_size * seq_len, device=pred.device), target_questions_flat]
        pred_selected = pred_selected.reshape(batch_size, seq_len)
        
        # Calculate binary cross entropy loss only on the valid positions
        valid_mask = (target_selectmasks == 1) & (target_questions != -1)
        
        if valid_mask.sum() > 0:
            loss = F.binary_cross_entropy_with_logits(
                pred_selected[valid_mask],
                target_responses[valid_mask]
            )
            
            self.log('train_loss', loss, prog_bar=True)
            return loss
        
        return None
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Batch of data from the data loader
            batch_idx: Index of the batch
        """
       
        questions = batch['questions']
        questions_embeddings = batch['question_embeddings']
        responses = batch['responses']
        selectmasks = batch['selectmasks']
        
        # Skip batches with too short sequences
        if questions.size(1) <= 1:
            return
            
        # Get model predictions
        if self.use_pretrained_embeddings:
            pred = self(questions_embeddings, responses, selectmasks)
        else:
            pred = self(questions, responses, selectmasks)
        
        # Calculate targets for all questions after the first step
        target_questions = questions[:, 1:]
        target_responses = responses[:, 1:].float()
        target_selectmasks = selectmasks[:, 1:]
        
        # Gather predictions for the target questions
        batch_size, seq_len = target_questions.size()
        # Ensure tensor is contiguous before reshaping
        pred_flat = pred.contiguous().reshape(batch_size * seq_len, -1)
        target_questions_flat = target_questions.contiguous().reshape(-1)
        
        # Handle out-of-bounds indices
        valid_indices = (target_questions_flat >= 0) & (target_questions_flat < self.num_questions)
        if not valid_indices.all():
            target_questions_flat = torch.clamp(target_questions_flat, 0, self.num_questions - 1)
            
        pred_selected = pred_flat[torch.arange(batch_size * seq_len, device=pred.device), target_questions_flat]
        pred_selected = pred_selected.reshape(batch_size, seq_len)
        
        # Calculate binary cross entropy loss only on the valid positions
        valid_mask = (target_selectmasks == 1) & (target_questions != -1)
        
        if valid_mask.sum() > 0:
            loss = F.binary_cross_entropy_with_logits(
                pred_selected[valid_mask],
                target_responses[valid_mask]
            )
            
            # Calculate AUC and accuracy
            # Apply sigmoid to get probabilities for metrics
            pred_probs = torch.sigmoid(pred_selected[valid_mask])
            pred_np = pred_probs.detach().cpu().numpy()
            target_np = target_responses[valid_mask].detach().cpu().numpy()
            
            try:
                auc = roc_auc_score(target_np, pred_np)
                acc = accuracy_score(target_np.round(), pred_np.round())
                
                self.log('val_loss', loss, prog_bar=True)
                self.log('val_auc', auc, prog_bar=True)
                self.log('val_acc', acc, prog_bar=True)
            except Exception as e:
                print(f"Error computing metrics: {e}")
                # Log default values to avoid breaking the training loop
                self.log('val_loss', loss, prog_bar=True)
                self.log('val_auc', 0.5, prog_bar=True)
                self.log('val_acc', 0.5, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Batch of data from the data loader
            batch_idx: Index of the batch
        """
        questions = batch['questions']
        responses = batch['responses']
        selectmasks = batch['selectmasks']
        questions_embeddings = batch['question_embeddings']
        # Skip batches with too short sequences
        if questions.size(1) <= 1:
            return
            
        # Get model predictions
        if self.use_pretrained_embeddings:
            pred = self(questions_embeddings, responses, selectmasks)
        else:
            pred = self(questions, responses, selectmasks)
        
        # Calculate targets for all questions after the first step
        target_questions = questions[:, 1:]
        target_responses = responses[:, 1:].float()
        target_selectmasks = selectmasks[:, 1:]
        
        # Gather predictions for the target questions
        batch_size, seq_len = target_questions.size()
        # Ensure tensor is contiguous before reshaping
        pred_flat = pred.contiguous().reshape(batch_size * seq_len, -1)
        target_questions_flat = target_questions.contiguous().reshape(-1)
        
        # Handle out-of-bounds indices
        valid_indices = (target_questions_flat >= 0) & (target_questions_flat < self.num_questions)
        if not valid_indices.all():
            target_questions_flat = torch.clamp(target_questions_flat, 0, self.num_questions - 1)
            
        pred_selected = pred_flat[torch.arange(batch_size * seq_len, device=pred.device), target_questions_flat]
        pred_selected = pred_selected.reshape(batch_size, seq_len)
        
        # Calculate binary cross entropy loss only on the valid positions
        valid_mask = (target_selectmasks == 1) & (target_questions != -1)
        
        if valid_mask.sum() > 0:
            loss = F.binary_cross_entropy_with_logits(
                pred_selected[valid_mask],
                target_responses[valid_mask]
            )
            
            # Calculate AUC and accuracy
            # Apply sigmoid to get probabilities for metrics
            pred_probs = torch.sigmoid(pred_selected[valid_mask])
            pred_np = pred_probs.detach().cpu().numpy()
            target_np = target_responses[valid_mask].detach().cpu().numpy()
            
            try:
                auc = roc_auc_score(target_np, pred_np)
                acc = accuracy_score(target_np.round(), pred_np.round())
                
                self.log('test_loss', loss)
                self.log('test_auc', auc)
                self.log('test_acc', acc)
            except Exception as e:
                print(f"Error computing test metrics: {e}")
                self.log('test_loss', loss)
                self.log('test_auc', 0.5)
                self.log('test_acc', 0.5)
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            dict: Optimizer and scheduler configuration
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]


class SAKTConfig:
    """Configuration for the SAKT model."""
    
    hidden_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class SAKT(pl.LightningModule):
    """Self-Attentive Knowledge Tracing model implementation."""
    
    def __init__(
        self, 
        num_questions: int,
        num_concepts: int,
        config: SAKTConfig = SAKTConfig(),
    ):
        """
        Initialize the SAKT model.
        
        Args:
            num_questions: Number of unique questions in the dataset
            num_concepts: Number of unique knowledge concepts in the dataset
            config: Configuration for the model
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        
        # Embedding for exercise+response
        self.exercise_embedding = nn.Embedding(
            num_embeddings=num_questions * 2,
            embedding_dim=self.hidden_dim
        )
        
        # Embedding for exercise to predict
        self.predict_embedding = nn.Embedding(
            num_embeddings=num_questions,
            embedding_dim=self.hidden_dim
        )
        
        # Position encoding
        self.position_embedding = nn.Embedding(
            num_embeddings=200,  # Maximum sequence length
            embedding_dim=self.hidden_dim
        )
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        
        # Output layer
        self.pred_layer = nn.Linear(self.hidden_dim, 1)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, questions, responses, selectmasks=None):
        """
        Forward pass through the model.
        
        Args:
            questions: Tensor of question IDs [batch_size, seq_len]
            responses: Tensor of response correctness [batch_size, seq_len]
            selectmasks: Tensor indicating which positions to select for prediction [batch_size, seq_len]
            
        Returns:
            pred: Tensor of logits [batch_size, seq_len]
        """
        batch_size, seq_len = questions.size()
        
        # Handle edge cases where sequence length is 1
        if seq_len <= 1:
            # Return empty tensor if sequence is too short
            return torch.zeros(batch_size, 0, device=self.device)
            
        # Create exercise+response embedding for the input sequence
        # We use the previous interactions to predict the next one
        exercise_ids = questions[:, :-1]
        exercise_responses = responses[:, :-1]
        
        # Clamp question IDs to valid range
        exercise_ids = torch.clamp(exercise_ids, 0, self.num_questions - 1)
        exercise_with_response = exercise_ids * 2 + exercise_responses
        
        # Target exercises to predict
        target_exercises = questions[:, 1:]
        target_exercises = torch.clamp(target_exercises, 0, self.num_questions - 1)
        
        # Create positional encoding
        positions = torch.arange(seq_len - 1, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed inputs
        exercise_emb = self.exercise_embedding(exercise_with_response)
        position_emb = self.position_embedding(positions)
        
        # Add positional encoding
        exercise_pos_emb = exercise_emb + position_emb
        
        # Embed target questions
        target_emb = self.predict_embedding(target_exercises)
        
        # Apply self-attention
        # First normalize
        norm_exercise_emb = self.layer_norm1(exercise_pos_emb)
        
        # Self-attention
        attn_output, _ = self.attention(
            query=target_emb,
            key=norm_exercise_emb,
            value=norm_exercise_emb
        )
        
        # Residual connection
        attn_output = attn_output + target_emb
        
        # Apply feed-forward network
        # First normalize
        norm_attn_output = self.layer_norm2(attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(norm_attn_output)
        
        # Residual connection
        output = ffn_output + attn_output
        
        # Output layer
        pred = self.pred_layer(output).squeeze(-1)
        
        # Return logits (no sigmoid - will be applied in loss function)
        return pred
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Batch of data from the data loader
            batch_idx: Index of the batch
            
        Returns:
            loss: Training loss
        """
        questions = batch['questions']
        responses = batch['responses']
        selectmasks = batch['selectmasks']
        
        # Skip batches with too short sequences
        if questions.size(1) <= 1:
            return None
            
        # Get model predictions
        pred = self(questions, responses, selectmasks)
        
        # Calculate targets for all questions after the first step
        target_questions = questions[:, 1:]
        target_responses = responses[:, 1:].float()
        target_selectmasks = selectmasks[:, 1:]
        
        # Calculate binary cross entropy loss only on the valid positions
        valid_mask = (target_selectmasks == 1) & (target_questions != -1)
        
        if valid_mask.sum() > 0:
            try:
                # Ensure tensors are the right shape and contiguous
                valid_pred = pred[valid_mask].contiguous()
                valid_target = target_responses[valid_mask].contiguous()
                
                loss = F.binary_cross_entropy_with_logits(valid_pred, valid_target)
                
                self.log('train_loss', loss, prog_bar=True)
                return loss
            except RuntimeError as e:
                print(f"Error in SAKT training step: {e}")
                return None
        
        return None
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Batch of data from the data loader
            batch_idx: Index of the batch
        """
        questions = batch['questions']
        responses = batch['responses']
        selectmasks = batch['selectmasks']
        
        # Skip batches with too short sequences
        if questions.size(1) <= 1:
            return
            
        # Get model predictions
        pred = self(questions, responses, selectmasks)
        
        # Calculate targets for all questions after the first step
        target_questions = questions[:, 1:]
        target_responses = responses[:, 1:].float()
        target_selectmasks = selectmasks[:, 1:]
        
        # Calculate binary cross entropy loss only on the valid positions
        valid_mask = (target_selectmasks == 1) & (target_questions != -1)
        
        if valid_mask.sum() > 0:
            try:
                # Ensure tensors are the right shape and contiguous
                valid_pred = pred[valid_mask].contiguous()
                valid_target = target_responses[valid_mask].contiguous()
                
                loss = F.binary_cross_entropy_with_logits(valid_pred, valid_target)
                
                # Calculate AUC and accuracy
                pred_probs = torch.sigmoid(valid_pred)
                pred_np = pred_probs.detach().cpu().numpy()
                target_np = valid_target.detach().cpu().numpy()
                
                auc = roc_auc_score(target_np, pred_np)
                acc = accuracy_score(target_np.round(), pred_np.round())
                
                self.log('val_loss', loss, prog_bar=True)
                self.log('val_auc', auc, prog_bar=True)
                self.log('val_acc', acc, prog_bar=True)
            except Exception as e:
                print(f"Error computing validation metrics: {e}")
                self.log('val_loss', loss, prog_bar=True)
                self.log('val_auc', 0.5, prog_bar=True)
                self.log('val_acc', 0.5, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Batch of data from the data loader
            batch_idx: Index of the batch
        """
        questions = batch['questions']
        responses = batch['responses']
        selectmasks = batch['selectmasks']
        
        # Skip batches with too short sequences
        if questions.size(1) <= 1:
            return
            
        # Get model predictions
        pred = self(questions, responses, selectmasks)
        
        # Calculate targets for all questions after the first step
        target_questions = questions[:, 1:]
        target_responses = responses[:, 1:].float()
        target_selectmasks = selectmasks[:, 1:]
        
        # Calculate binary cross entropy loss only on the valid positions
        valid_mask = (target_selectmasks == 1) & (target_questions != -1)
        
        if valid_mask.sum() > 0:
            try:
                # Ensure tensors are the right shape and contiguous
                valid_pred = pred[valid_mask].contiguous()
                valid_target = target_responses[valid_mask].contiguous()
                
                loss = F.binary_cross_entropy_with_logits(valid_pred, valid_target)
                
                # Calculate AUC and accuracy
                pred_probs = torch.sigmoid(valid_pred)
                pred_np = pred_probs.detach().cpu().numpy()
                target_np = valid_target.detach().cpu().numpy()
                
                auc = roc_auc_score(target_np, pred_np)
                acc = accuracy_score(target_np.round(), pred_np.round())
                
                self.log('test_loss', loss)
                self.log('test_auc', auc)
                self.log('test_acc', acc)
            except Exception as e:
                print(f"Error computing test metrics: {e}")
                self.log('test_loss', loss)
                self.log('test_auc', 0.5)
                self.log('test_acc', 0.5)
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            dict: Optimizer and scheduler configuration
        """
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer], [scheduler] 