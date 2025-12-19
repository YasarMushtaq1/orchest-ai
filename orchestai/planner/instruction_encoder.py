"""
Instruction Encoder: Transformer-based encoder for processing natural language goals
"""

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer, BertModel, BertTokenizer
from typing import Dict, Optional


class InstructionEncoder(nn.Module):
    """
    Transformer-based encoder for processing natural language instructions.
    Supports T5 and BERT architectures.
    """
    
    def __init__(
        self,
        model_name: str = "t5-base",
        hidden_size: int = 768,
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # Initialize base transformer model
        if "t5" in model_name.lower():
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            # T5 uses relative position embeddings, get hidden size from config
            self.hidden_size = self.encoder.config.d_model
        elif "bert" in model_name.lower():
            self.encoder = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.hidden_size = self.encoder.config.hidden_size
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Projection layer to standardize output dimension
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
    def forward(
        self,
        instructions: list[str],
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode natural language instructions into dense representations.
        
        Args:
            instructions: List of natural language instruction strings
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - embeddings: [batch_size, hidden_size] instruction embeddings
                - pooled: [batch_size, hidden_size] pooled representation
                - attention: Optional attention weights if return_attention=True
        """
        # Tokenize instructions
        encoded = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Move to device
        device = next(self.encoder.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Encode
        if "t5" in self.model_name.lower():
            outputs = self.encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True,
            )
            # Use mean pooling over sequence length for T5
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            hidden_states = outputs.last_hidden_state
        else:  # BERT
            outputs = self.encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True,
            )
            pooled = outputs.pooler_output
            hidden_states = outputs.last_hidden_state
        
        # Project to standard hidden size
        pooled_projected = self.projection(pooled)
        
        result = {
            "embeddings": pooled_projected,
            "pooled": pooled_projected,
            "hidden_states": hidden_states,
        }
        
        if return_attention and hasattr(outputs, "attentions"):
            result["attention"] = outputs.attentions
        
        return result
    
    def encode(self, instruction: str) -> torch.Tensor:
        """
        Convenience method for encoding a single instruction.
        
        Args:
            instruction: Single instruction string
            
        Returns:
            [hidden_size] instruction embedding
        """
        self.eval()
        with torch.no_grad():
            result = self.forward([instruction])
            return result["embeddings"].squeeze(0)

