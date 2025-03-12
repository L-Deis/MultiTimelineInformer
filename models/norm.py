import torch
import torch.nn as nn

# Conditional Layer Normalization module: from 
class ConditionalLayerNorm(nn.Module):
    def __init__(self, normalized_shape, condition_dim, eps=1e-5):
        """
        Args:
            normalized_shape (int): The size of the feature dimension to normalize (e.g., d_model).
            condition_dim (int): The dimensionality of the static condition (e.g., size of c).
            eps (float): Small value to avoid division by zero in normalization.
        """
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        
        # Base normalization parameters (learnable, like in standard LayerNorm)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        # MLP to generate scale and bias from the condition
        self.condition_to_scale = nn.Linear(condition_dim, normalized_shape)
        self.condition_to_bias = nn.Linear(condition_dim, normalized_shape)

    def forward(self, x, condition):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].
            condition (torch.Tensor): Condition tensor of shape [batch_size, condition_dim].
        Returns:
            torch.Tensor: Normalized and conditioned output, same shape as x.
        """
        # Compute mean and variance along the feature dimension
        mean = x.mean(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [batch_size, seq_len, 1]
        
        # Standard normalization
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # [batch_size, seq_len, d_model]
        
        # Generate condition-dependent scale and bias
        scale = self.weight + self.condition_to_scale(condition)  # [batch_size, d_model]
        bias = self.bias + self.condition_to_bias(condition)     # [batch_size, d_model]
        
        # Reshape scale and bias to match input dimensions
        scale = scale.unsqueeze(1)  # [batch_size, 1, d_model]
        bias = bias.unsqueeze(1)    # [batch_size, 1, d_model]
        
        # Apply conditional normalization
        return scale * x_norm + bias
    