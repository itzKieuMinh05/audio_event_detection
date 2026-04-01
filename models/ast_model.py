"""
Audio Spectrogram Transformer (AST) Model
Based on Vision Transformer architecture adapted for audio spectrograms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import yaml
import math


class PatchEmbedding(nn.Module):
    """
    Convert spectrogram into patches and embed them
    """
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (128, 400),
                 patch_size: Tuple[int, int] = (16, 16),
                 embed_dim: int = 768):
        """
        Initialize patch embedding
        
        Args:
            input_size: (n_mels, time_frames)
            patch_size: Size of each patch
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.n_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])
        
        # Patch embedding layer
        self.projection = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, 1, n_mels, time_frames)
            
        Returns:
            Patch embeddings (batch, n_patches, embed_dim)
        """
        # x: (B, 1, H, W) -> (B, embed_dim, H', W')
        x = self.projection(x)
        
        # Flatten patches
        # (B, embed_dim, H', W') -> (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Attention output (batch, seq_len, embed_dim)
            (Optional) Attention weights (batch, num_heads, seq_len, seq_len)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_weights = attn  # Keep original weights before dropout for visualization
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        if return_attention:
            return x, attn_weights
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron (feed-forward network)
    """
    
    def __init__(self, 
                 in_features: int, 
                 hidden_features: Optional[int] = None,
                 dropout: float = 0.1):
        """
        Initialize MLP
        
        Args:
            in_features: Input dimension
            hidden_features: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1):
        """
        Initialize transformer block
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class AudioSpectrogramTransformer(nn.Module):
    """
    Audio Spectrogram Transformer (AST) for audio event detection
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize AST model
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.model_config = config['model']['ast']
        self.num_classes = config['model']['num_classes']
        
        # Model parameters
        input_size = tuple(self.model_config['input_size'])
        patch_size = tuple(self.model_config['patch_size'])
        embed_dim = self.model_config['embed_dim']
        depth = self.model_config['depth']
        num_heads = self.model_config['num_heads']
        mlp_ratio = self.model_config['mlp_ratio']
        dropout = self.model_config['dropout']
        attention_dropout = self.model_config['attention_dropout']
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(input_size, patch_size, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attention_dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, self.num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other layers
        self.apply(self._init_layer_weights)
    
    def _init_layer_weights(self, m):
        """Initialize layer weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input spectrogram (batch, 1, n_mels, time_frames)
            
        Returns:
            Class logits (batch, num_classes)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add position embedding (resize if input patch count differs)
        if self.pos_embed.size(1) != x.size(1):
            cls_pos = self.pos_embed[:, :1, :]
            patch_pos = self.pos_embed[:, 1:, :]

            # original grid (height, width) from configured input/patch sizes
            old_h = self.patch_embed.input_size[0] // self.patch_embed.patch_size[0]
            old_w = self.patch_embed.input_size[1] // self.patch_embed.patch_size[1]

            new_n = x.size(1) - 1
            new_h = old_h
            new_w = int(new_n // new_h)

            D = patch_pos.size(2)

            # reshape to (1, D, old_h, old_w) -> interpolate -> reshape back
            patch_pos = patch_pos.reshape(1, old_h, old_w, D).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_h, new_w), mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, D)

            pos_embed = torch.cat([cls_pos, patch_pos], dim=1)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed.to(x.device)
        x = self.pos_drop(x)
        
        # Transformer encoder
        for block in self.blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Classification head (use class token)
        x = x[:, 0]
        x = self.head(x)
        
        return x
    
    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Get attention maps for visualization
        
        Args:
            x: Input spectrogram
            
        Returns:
            List of attention maps from each layer
        """
        attention_maps = []
        
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # Add (possibly resized) position embedding for attention map extraction
        if self.pos_embed.size(1) != x.size(1):
            cls_pos = self.pos_embed[:, :1, :]
            patch_pos = self.pos_embed[:, 1:, :]
            old_h = self.patch_embed.input_size[0] // self.patch_embed.patch_size[0]
            old_w = self.patch_embed.input_size[1] // self.patch_embed.patch_size[1]
            new_n = x.size(1) - 1
            new_h = old_h
            new_w = int(new_n // new_h)
            D = patch_pos.size(2)
            patch_pos = patch_pos.reshape(1, old_h, old_w, D).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_h, new_w), mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, D)
            pos_embed = torch.cat([cls_pos, patch_pos], dim=1)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed.to(x.device)
        x = self.pos_drop(x)
        
        for block in self.blocks:
            # Extract attention weights properly
            attn_output, attn_weights = block.attn(block.norm1(x), return_attention=True)
            attention_maps.append(attn_weights)  # Append real attention weights
            x = x + attn_output
            x = x + block.mlp(block.norm2(x))
        
        return attention_maps


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model architecture"""
    print("Testing Audio Spectrogram Transformer...")
    
    # Create model
    model = AudioSpectrogramTransformer(config_path="h:/audio_event_detection/configs/config.yaml")
    
    # Print model info
    print(f"Number of parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 128, 400)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("Model test complete!")


if __name__ == "__main__":
    test_model()
