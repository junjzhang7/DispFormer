import math
from typing import Any, Dict

import torch
from einops import rearrange
from torch import Tensor, nn

from .Attention import MultiheadAttention
from .layers import MultiheadAttentionBlock


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        max_len: int = 1000,
        dropout=0.1,
        positional_encoding_type="sincos",
    ):
        super().__init__()
        # postional encoding: [len x d_model]
        self.position_enc = self._init_pe(d_model, max_len, positional_encoding_type)
        # Positional dropout
        self.positional_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _init_pe(
        d_model: int, max_len: int, positional_encoding_type: str
    ) -> nn.Parameter:
        # Positional encoding
        if positional_encoding_type == "random":
            position_enc = nn.Parameter(
                torch.randn(max_len, d_model), requires_grad=True
            )
        elif positional_encoding_type == "sincos":
            position_enc = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
            )
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            raise ValueError(
                f"{positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'."
            )
        return position_enc

    def forward(self, x: torch.Tensor):
        # x: [bxc, l, d] or [b, c, l, d]
        len = x.size(-2)
        hidden_state = self.positional_dropout(x + self.position_enc[:len])
        return hidden_state


class TimeEmbedder(nn.Module):
    def __init__(self, time_emb_dim, n_channels):
        super().__init__()
        self.periodic = nn.Linear(1, time_emb_dim - 1)
        self.non_periodic = nn.Linear(1, 1)
        self.k_map = nn.Parameter(torch.ones(1, n_channels, 1, time_emb_dim))

    def forward(self, time):
        """
        Args:
            time (_type_): [bs, len]

        Returns:
            _type_: [bs, n_channels, len, dim]
        """
        time = rearrange(time, "b l -> b 1 l 1")
        out2 = torch.sin(self.periodic(time))
        out1 = self.non_periodic(time)
        out = torch.cat([out1, out2], -1)  # [b, 1, l, d]
        out = torch.mul(out, self.k_map)  # [b, c, l, d]
        return out


class ValueEmbedder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embedder = nn.Linear(1, d_model)

    def forward(self, value, indicator=None):
        """
        Args:
            value (_type_): [b, c, l]
            indicator:  [b, c, l]
        Returns:
            _type_: [b, c, l, d]
        """
        embedding = self.embedder(value)
        if indicator is not None:
            embedding = embedding * indicator
        return embedding


class ChannelEmbedder(nn.Module):
    def __init__(self, d_model, n_channels):
        super().__init__()
        self.channel_embedder = nn.Embedding(n_channels + 1, d_model, padding_idx=0)

    def forward(self, type_matrix):
        channel_embedding = self.channel_embedder(type_matrix.long())
        return channel_embedding


class IndicatorEmbedder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = nn.Linear(1, d_model)

    def forward(self, x):
        """
        Args:
            x (_type_): [b, c, l]

        Returns:
            _type_: [b, c, l, d]
        """
        x = self.encoder(x.unsqueeze(-1))
        return x


class DeltaEmbedder(nn.Module):
    def __init__(self, d_model, n_channels, inter_dim=16):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(1, inter_dim),
            nn.Tanh(),
            nn.Linear(inter_dim, d_model, bias=False),
        )
        self.k_map = nn.Parameter(torch.ones(1, n_channels, 1, d_model))

    def forward(self, delta, indicator=None):
        embedding = self.embedder(delta)
        embedding = torch.mul(embedding, self.k_map)
        if indicator is not None:
            embedding = embedding * indicator
        return embedding


class WarpformerEmbedder(nn.Module):
    def __init__(self, d_model, n_channels, dropout=0.1) -> None:
        super().__init__()
        self.time_embedder = TimeEmbedder(d_model, n_channels)
        self.value_embedder = ValueEmbedder(d_model)
        self.channel_embedder = ChannelEmbedder(d_model, n_channels)
        self.delta_embedder = DeltaEmbedder(d_model, n_channels)

        self.type_matrix = torch.tensor([int(i) for i in range(1, n_channels + 1)])
        self.type_matrix = rearrange(self.type_matrix, "c -> 1 c 1")

        self.dropout = nn.Dropout(dropout)

    def forward(self, time, delta, value, indicator):
        value = rearrange(value, "b l c -> b c l 1")
        indicator = rearrange(indicator, "b l c -> b c l 1")
        delta = rearrange(delta, "b l c -> b c l 1")

        value_embedding = self.value_embedder(value, indicator)
        time_embedding = self.time_embedder(time)
        delta_embedding = self.delta_embedder(delta, indicator)
        channel_embedding = self.channel_embedder(self.type_matrix.to(value.device))

        embedding = (
            value_embedding + time_embedding + delta_embedding + channel_embedding
        )

        embedding = self.dropout(embedding)
        return embedding


class TPatchEmbedder(nn.Module):
    def __init__(self, d_model, time_emb_dim=10) -> None:
        super().__init__()
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, time_emb_dim - 1)

        ttcn_dim = d_model - 1
        input_dim = time_emb_dim + 1  # 11

        self.ttcn_dim = ttcn_dim
        self.filter_generators = nn.Sequential(
            nn.Linear(input_dim, ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ttcn_dim, ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ttcn_dim, input_dim * ttcn_dim, bias=True),
        )
        self.T_bias = nn.Parameter(torch.randn(1, ttcn_dim))

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def TTCN(self, x, mask_X):
        N, Lx, _ = mask_X.size()

        filter = self.filter_generators(x)
        filter_mask = filter * mask_X + (1 - mask_X) * (-1e8)
        # normalize along with sequence dimension
        Filter_seqnorm = nn.functional.softmax(
            filter_mask, dim=-2
        )  # (N, Lx, F_in*ttcn_dim)
        Filter_seqnorm = Filter_seqnorm.view(
            N, Lx, self.ttcn_dim, -1
        )  # (N, Lx, ttcn_dim, F_in)
        X_int_broad = x.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(
            torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1
        )  # (N, ttcn_dim)
        h_t = torch.relu(ttcn_out + self.T_bias)  # (N, ttcn_dim)
        return h_t

    def forward(self, time, delta, X, mask_X):
        batch_size, n_patch, len_patch, n_channel = X.size()
        X = rearrange(X, "b np lp c -> (b np c) lp 1")
        mask_X = rearrange(mask_X, "b np lp c -> (b np c) lp 1")

        time = rearrange(time, "b np lp c -> (b np c) lp 1")
        time_embedding = self.LearnableTE(time)  # [(b np c), lp, 10]
        X = torch.cat([X, time_embedding], -1)

        # mask for the patch
        mask_patch = mask_X.sum(dim=1) > 0  # [b x np x c, 1]
        ### TTCN for patch modeling ###
        x_patch = self.TTCN(X, mask_X)  # [b x np x c, d-1]
        x_patch = torch.cat([x_patch, mask_patch], dim=-1)  # [b x np x c, d]
        x_patch = rearrange(x_patch, "(b np c) d -> b c np d", b=batch_size, np=n_patch)
        return x_patch


class LinearEmbedder(nn.Module):
    def __init__(self, d_model, n_channels, dropout=0.1) -> None:
        super().__init__()
        self.time_embedder = TimeEmbedder(d_model, n_channels)
        self.value_embedder = nn.Linear(1, d_model)
        self.indicator_embedder = nn.Linear(1, d_model)
        self.delta_embedder = nn.Linear(1, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, time, delta, value, indicator):
        """
        Args:
            time (_type_): [bs, len]
            tau (_type_): [bs, len, n_channels]
            value (_type_): [bs, len, n_channels]
            indicator (_type_): [bs, len, n_channels]

        Return:
            embedding: [bs, n_channels, len, dim]
        """

        value = rearrange(value, "b l c -> b c l 1")
        indicator = rearrange(indicator, "b l c -> b c l 1")
        delta = rearrange(delta, "b l c -> b c l 1")

        time_embedding = self.time_embedder(time)
        delta_embedding = self.delta_embedder(delta)
        value_embedding = self.value_embedder(value)
        indicator_embedding = self.indicator_embedder(indicator)

        embedding = (
            time_embedding + delta_embedding + value_embedding + indicator_embedding
        )

        embedding = self.dropout(embedding)
        return embedding


class Conv1dEmbedder(nn.Module):
    def __init__(self, args, d_model, n_channels, dropout=0.1, conv_bias=False) -> None:
        super().__init__()
        self.args = args

        self.time_embedder = TimeEmbedder(d_model, n_channels)

        self.value_embedder = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=conv_bias,
        )
        self.indicator_embedder = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=conv_bias,
        )
        self.delta_embedder = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=conv_bias,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

        self.dropout = nn.Dropout(dropout)

    def forward(self, time, delta, value, indicator):
        """
        Args:
            time (_type_): [bs, len]
            tau (_type_): [bs, len, n_channels]
            value (_type_): [bs, len, n_channels]
            indicator (_type_): [bs, len, n_channels]

        Return:
            embedding: [bs, n_channels, len, dim]
        """
        bs = value.size(0)
        value = rearrange(value, "b l c -> (b c) 1 l")
        indicator = rearrange(indicator, "b l c -> (b c) 1 l")
        delta = rearrange(delta, "b l c -> (b c) 1 l")

        value_embedding = self.value_embedder(value)
        value_embedding = rearrange(value_embedding, "(b c) d l -> b c l d", b=bs)
        indicator_embedding = self.indicator_embedder(indicator)
        indicator_embedding = rearrange(
            indicator_embedding, "(b c) d l -> b c l d", b=bs
        )
        time_embedding = self.time_embedder(time)
        # delta_embedding = self.delta_embedder(delta)

        embedding = value_embedding + time_embedding + indicator_embedding

        embedding = self.dropout(embedding)
        return embedding


class Conv2dEmbedder(nn.Module):
    def __init__(
        self, d_model, n_channels, c_patch_size=4, t_patch_size=4, dropout=0.1
    ) -> None:
        super().__init__()
        self.time_embedder = TimeEmbedder(d_model, n_channels)

        self.value_embedder = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_patch_size, t_patch_size),
            stride=(c_patch_size, t_patch_size),
        )
        self.indicator_embedder = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_patch_size, t_patch_size),
            stride=(c_patch_size, t_patch_size),
        )
        self.delta_embedder = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_patch_size, t_patch_size),
            stride=(c_patch_size, t_patch_size),
        )
        self.c_patch_size = c_patch_size
        self.t_patch_size = t_patch_size

        self.dropout = nn.Dropout(dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, time, delta, value, indicator):
        """
        Args:
            time (_type_): [bs, len]
            tau (_type_): [bs, len, n_channels]
            value (_type_): [bs, len, n_channels]
            indicator (_type_): [bs, len, n_channels]

        Return:
            embedding: [bs, n_channels, len, dim]
        """
        seq_len = value.size(1)
        n_channels = value.size(2)
        if seq_len % self.t_patch_size != 0:
            t_pad_num = (seq_len // self.t_patch_size + 1) * self.t_patch_size - seq_len
        else:
            t_pad_num = 0
        if n_channels % self.c_patch_size != 0:
            c_pad_num = (
                n_channels // self.c_patch_size + 1
            ) * self.c_patch_size - n_channels
        else:
            c_pad_num = 0

        value = rearrange(value, "b l c -> b 1 c l")
        indicator = rearrange(indicator, "b l c -> b 1 c l")
        delta = rearrange(delta, "b l c -> b 1 c l")

        value = nn.functional.pad(
            value, pad=(0, t_pad_num, 0, c_pad_num), mode="constant", value=0
        )
        indicator = nn.functional.pad(
            indicator, pad=(0, t_pad_num, 0, c_pad_num), mode="constant", value=0
        )
        delta = nn.functional.pad(
            delta, pad=(0, t_pad_num, 0, c_pad_num), mode="constant", value=0
        )

        # value_embedding: [b, d, c_np, t_np]
        value_embedding = self.value_embedder(value)
        # value_embedding: [b, c_npxt_np, d]
        value_embedding = value_embedding.flatten(2).transpose(1, 2)

        indicator_embedding = self.indicator_embedder(indicator)
        indicator_embedding = indicator_embedding.flatten(2).transpose(1, 2)

        delta_embedding = self.delta_embedder(indicator)
        delta_embedding = delta_embedding.flatten(2).transpose(1, 2)

        embedding = value_embedding + indicator_embedding + delta_embedding

        embedding = self.dropout(embedding)
        return embedding


class MultiScaleAgg(nn.Module):
    def __init__(self, d_model, n_scales=2, dropout=0.1) -> None:
        super().__init__()
        self.n_scales = n_scales

        self.multi_scale_agg = nn.ModuleList(
            [
                MultiheadAttentionBlock(
                    MultiheadAttention(d_model, n_heads=4),
                    d_model=d_model,
                    dropout=dropout,
                )
                for i in range(n_scales - 1)
            ]
        )
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for i in range(n_scales - 1)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for i in range(n_scales - 1)])
        self.dropout_path = nn.ModuleList(
            [nn.Dropout(dropout) for i in range(n_scales - 1)]
        )

    def forward(self, embedding_list):
        "embedding_list: [[b, c, l, d],....]"
        batch_size = embedding_list[0].size(0)

        for i in range(len(embedding_list) - 1):

            q = self.norm1[i](rearrange(embedding_list[i + 1], "b c l d -> (b c) l d"))
            k = self.norm2[i](rearrange(embedding_list[i], "b c l d -> (b c) l d"))
            v = self.norm2[i](rearrange(embedding_list[i], "b c l d -> (b c) l d"))

            attention_out = rearrange(
                self.multi_scale_agg[i](q, k, v), "(b c) l d -> b c l d", b=batch_size
            )

            embedding_list[i + 1] = embedding_list[i + 1] + self.dropout_path[i](
                attention_out
            )

        hidden = embedding_list[-1]
        return hidden


class MultiScaleEmbedder(nn.Module):
    def __init__(
        self,
        d_model,
        n_channels,
        n_scales=3,
        down_sampling_window=2,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.down_sampling_window = down_sampling_window

        self.time_embedder = TimeEmbedder(d_model, n_channels)
        self.indicator_embedder = nn.Linear(1, d_model)
        self.value_embedders = nn.ModuleList(
            nn.Linear(1, d_model) for i in range(self.n_scales)
        )
        self.multi_scale_agg = MultiScaleAgg(d_model, n_scales)
        self.dropout = nn.Dropout(dropout)

    def forward(self, time, delta, value, indicator):
        """
        Args:
            time (_type_): [bs, len]
            tau (_type_): [bs, len, n_channels]
            value (_type_): [bs, len, n_channels]
            indicator (_type_): [bs, len, n_channels]

        Return:
            embedding: [bs, n_channels, len, dim]
        """
        value = rearrange(value, "b l c -> b c l")
        indicator = rearrange(indicator, "b l c -> b c l")

        downsample = nn.AvgPool1d(kernel_size=self.down_sampling_window).to(
            value.device
        )

        value_embedding_list = []
        value_embedding_list.append(self.value_embedders[-1](value.unsqueeze(-1)))

        for i in range(self.n_scales - 1):
            x_sample = downsample(value)
            emb = self.value_embedders[i](x_sample.unsqueeze(-1))
            value_embedding_list.append(emb)
            value = x_sample

        value_embedding_list = list(reversed(value_embedding_list))
        value_embedding = self.multi_scale_agg(value_embedding_list)  # [b, c, l, d]

        time_embedding = self.time_embedder(time)
        indicator_embedding = self.indicator_embedder(indicator.unsqueeze(-1))

        embedding = value_embedding + time_embedding + indicator_embedding

        embedding = self.dropout(embedding)
        return embedding


class GatedFusion(nn.Module):
    def __init__(self, d_model, dropout=0.1) -> None:
        super().__init__()
        self.gate_w1 = nn.Linear(d_model, d_model)
        self.gate_w2 = nn.Linear(d_model, d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_1, x_2):
        gate = torch.sigmoid(self.gate_w1(x_1) + self.gate_w2(x_2))
        x = gate * x_1 + (1 - gate) * x_2

        x = self.dropout(self.projection(x))
        return x


if __name__ == "__main__":
    time = torch.rand([1, 50])
    delta = torch.randn([1, 50, 36])
    value = torch.randn([1, 50, 36])
    indicator = torch.randn([1, 50, 36])

    # embedder = PatchEmbedder(d_model=64)
    # x = embedder(time, delta, value, indicator)
