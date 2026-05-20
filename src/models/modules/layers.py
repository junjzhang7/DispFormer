from typing import Optional

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        dropout: float = 0.1,
        qkv_bias=True,
        flash=True,
    ):
        super().__init__()
        self.flash = flash
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = d_model // n_heads
        self.scaling = self.head_dim**-0.5
        self.qkv_bias = qkv_bias

        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

        self.out_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

        if self.qkv_bias:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        hidden_states: [b, l, d]
        """
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_states is not None and value_states is not None

        # get query proj: [bs, len, dim]
        q_states = self.q_proj(hidden_states)
        q_states = rearrange(q_states, "b l (h d) -> b h l d", h=self.n_heads)

        if is_cross_attention:
            # cross_attentions
            k_states = self.k_proj(key_states)
            v_states = self.v_proj(value_states)
        else:
            # self_attention
            k_states = self.k_proj(hidden_states)
            v_states = self.v_proj(hidden_states)

        k_states = rearrange(k_states, "b l (h d) -> b h l d", h=self.n_heads)
        v_states = rearrange(v_states, "b l (h d) -> b h l d", h=self.n_heads)

        if attention_mask is not None:
            # attention_mask: [bs, n_heads, len, len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        if self.flash:
            attn_output = nn.functional.scaled_dot_product_attention(
                q_states,
                k_states,
                v_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            attn_weights = None
        else:
            # attn_weights: [bs, n_heads, len, len]
            attn_weights = (
                torch.matmul(q_states, k_states.transpose(-1, -2))
            ) * self.scaling

            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            attn_probs = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

            # [bs, n_heads, len, head_dim]
            attn_output = torch.matmul(attn_probs, v_states)

        attn_output = rearrange(attn_output, "b h l d -> b l (h d)")

        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class MultiheadAttentionBlock(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        dropout=0.1,
        norm_layer=nn.LayerNorm,
        norm_first=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(d_model)
        self.norm_first = norm_first
        self.attn = attention
        self.dropout_path1 = nn.Dropout(dropout)

    def forward(
        self,
        hidden,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # attention & norm
        if self.norm_first:
            attn_output, attn_weights = self.attn(
                self.norm1(hidden), key_states, value_states, attention_mask
            )
            hidden = hidden + self.dropout_path1(attn_output)
        else:
            attn_output, attn_weights = self.attn(
                hidden, key_states, value_states, attention_mask
            )
            hidden = self.norm1(hidden + self.dropout_path1(attn_output))
        return hidden, attn_weights


class FFBlock(nn.Module):
    def __init__(
        self,
        d_model,
        ffn_dim,
        act_layer=nn.GELU,
        dropout=0.1,
        norm_first=True,
        conv_ff=False,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            act_layer(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, d_model),
        )
        if conv_ff:
            self.ff = nn.Sequential(
                Rearrange("b c l d -> (b c) d l"),
                nn.Conv1d(in_channels=d_model, out_channels=ffn_dim, kernel_size=1),
                act_layer(),
                nn.Dropout(dropout),
                nn.Conv1d(in_channels=ffn_dim, out_channels=d_model, kernel_size=1),
                Rearrange("(b c) d l -> b c l d", c=36),
            )

    def forward(self, hidden):
        if self.norm_first:
            hidden = hidden + self.dropout(self.ff(self.norm(hidden)))
        else:
            hidden = self.norm(hidden + self.dropout(self.ff(hidden)))
        return hidden
