from typing import Any, Dict, Optional

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from ..base_model import BaseModel
from ..modules.layers import MultiheadAttention
from ..modules.embedder import PositionalEncoding
from ..modules.layers import FFBlock, MultiheadAttentionBlock


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


class TimeEmbedder(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()
        self.periodic = nn.Linear(1, time_emb_dim - 1)
        self.non_periodic = nn.Linear(1, 1)

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
        return out



class LinearEmbedder(nn.Module):
    def __init__(self, args, d_model, dropout=0.1) -> None:
        super().__init__()
        self.args = args
        self.time_embedder = TimeEmbedder(d_model)

        if self.args.wo_delta or self.args.wo_indicator:
            self.value_embedder = nn.Sequential(
                nn.Linear(2, d_model),
                nn.Linear(d_model, d_model),
            )
        else:
            self.value_embedder = nn.Sequential(
                nn.Linear(3, d_model),
                nn.Linear(d_model, d_model),
            )

        self.norm = nn.LayerNorm(d_model)
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

        if self.args.wo_delta:
            x = torch.cat([value, indicator], -1)
        elif self.args.wo_indicator:
            x = torch.cat([value, delta], -1)
        else:
            x = torch.cat([value, indicator, delta], -1)

        value_embedding = self.value_embedder(x)  # [b, c, l, d]

        if self.args.wo_time:
            embedding = value_embedding
        else:
            embedding = time_embedding + value_embedding  # + var_emb

        embedding = self.dropout(self.norm(embedding))
        return embedding


class DualAttentionBlock(nn.Module):
    def __init__(
        self,
        args,
        d_model,
        n_heads,
        ffn_dim,
        qkv_bias=False,
        dropout=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_first=True,
        n_channels=36,
        flash=True,
    ):
        super().__init__()
        self.args = args

        self.seq_att = MultiheadAttention(
            d_model, n_heads, dropout, qkv_bias, flash=flash
        )
        self.seq_att_block = MultiheadAttentionBlock(
            attention=self.seq_att,
            d_model=d_model,
            dropout=dropout,
            norm_layer=norm_layer,
            norm_first=norm_first,
        )

        self.var_att = MultiheadAttention(
            d_model, n_heads, dropout, qkv_bias, flash=flash
        )
        self.var_att_block = MultiheadAttentionBlock(
            attention=self.var_att,
            d_model=d_model,
            dropout=dropout,
            norm_layer=norm_layer,
            norm_first=norm_first,
        )

        self.ff_block = FFBlock(
            d_model=d_model,
            ffn_dim=ffn_dim,
            act_layer=act_layer,
            dropout=dropout,
            norm_first=norm_first,
        )

        self.dispatcher = nn.Parameter(torch.randn([n_channels, d_model]))

        self.proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.gate_fusion = GatedFusion(d_model, dropout)

    def forward(self, hidden):

        if self.args.vanilla_dual_att:
            return self.vanilla_dual_att(hidden)
        else:
            return self.global_dual_att(hidden)

    def global_dual_att(self, hidden):
        # hidden:[b, c, l, d]
        b = hidden.size(0)
        c = hidden.size(1)

        hidden = rearrange(hidden, "b c l d -> (b c) l d")

        dispatchers = repeat(self.dispatcher, "c d -> (b c) 1 d", b=b)
        hidden_expand = torch.cat([dispatchers, hidden], 1)

        hidden_expand, seq_attn_weights = self.seq_att_block(hidden_expand)

        # channel interaction
        dispatchers = hidden_expand[:, 0]  # [bxc, d]
        dispatchers, var_attn_weights = self.var_att_block(
            rearrange(dispatchers, "(b c) d -> b c d", c=c)
        )
        hidden = hidden_expand[:, 1:]
        hidden = self.distribute(dispatchers, hidden)

        hidden = self.ff_block(hidden)

        hidden = rearrange(hidden, "(b c) l d -> b c l d", c=c)
        return hidden

    def distribute(self, dispatchers, hidden):
        # dispatchers: [b, c, d]  hidden: [bxc, l, d]
        if self.args.distribute_style == "add":
            hidden = rearrange(dispatchers, "b c d -> (b c) 1 d") + hidden

        elif self.args.distribute_style == "concat":
            dispatchers = repeat(dispatchers, "b c d -> (b c) l d", l=hidden.size(1))
            hidden = self.proj(torch.cat([dispatchers, hidden], -1))

        elif self.args.distribute_style == "gate":
            dispatchers = repeat(dispatchers, "b c d -> (b c) l d", l=hidden.size(1))
            hidden = self.gate_fusion(dispatchers, hidden)
        return hidden

    def vanilla_dual_att(self, hidden):
        # hidden:[b, c, l, d]
        b = hidden.size(0)

        hidden = rearrange(hidden, "b c l d -> (b c) l d")
        hidden = self.seq_att_block(hidden)

        # channel interaction
        hidden = rearrange(hidden, "(b c) l d -> (b l) c d", b=b)
        hidden = self.var_att_block(hidden)
        hidden = rearrange(hidden, "(b l) c d -> (b c) l d", b=b)

        hidden = self.ff_block(hidden)

        hidden = rearrange(hidden, "(b c) l d -> b c l d", b=b)
        return hidden


class Encoder(nn.Module):
    def __init__(
        self, args, n_layers, d_model, n_heads, n_channels, dropout=0.1
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                DualAttentionBlock(
                    args,
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_dim=args.ff_expand_factor * d_model,
                    qkv_bias=args.qkv_bias,
                    dropout=dropout,
                    norm_first=args.norm_first,
                    n_channels=n_channels,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden):
        # hidden:[b, c, l, d]
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        return hidden


class Bottleneck(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.init_model()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, hidden):
        assignment = self.compressor(hidden)  # [bxc, l, d]
        mask = torch.sigmoid(assignment)
        hidden = mask * hidden
        return hidden


class DispFormer(BaseModel):
    def __init__(self, args, d_model, n_layers, n_heads, dropout=0.1):
        super().__init__(args)
        self.embedder = LinearEmbedder(args, d_model, dropout)
        self.positional_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.encoder = Encoder(
            args,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            n_channels=self.n_channels,
            dropout=dropout,
        )
        self.bottleneck = Bottleneck(d_model)

        self.head = nn.Linear(self.n_channels * d_model, self.n_classes)
        self.head_org = nn.Linear(self.n_channels * d_model, self.n_classes)

        self.save_hyperparameters()

    def forward(self, time, delta, value, indicator):
        embedding = self.embedder(time, delta, value, indicator)  # [b, c, l, d]

        embedding = self.positional_encoder(embedding)

        hidden, layer_seq_att_score, layer_var_att_score = self.encoder(
            embedding
        )  # [b, c, l, d]

        hidden_ib = self.bottleneck(rearrange(hidden, "b c l d -> (b c) l d"))
        hidden_ib = rearrange(hidden_ib, "(b c) l d -> b c l d", c=self.n_channels)

        logits = self.head(rearrange(hidden_ib.mean(2), "b c d -> b (c d)"))
        return {"logits": logits, "hidden": hidden, "hidden_ib": hidden_ib}

    def training_step(self, batch, batch_idx) -> Tensor | Dict[str, Any]:
        time = batch["time"]
        value = batch["value"]
        indicator = batch["indicator"]
        delta = batch["delta"]
        label = batch["label"]

        outputs = self.forward(time, delta, value, indicator)
        logits = outputs["logits"]
        ce_loss = nn.CrossEntropyLoss()(logits, label)

        logits_org = self.head_org(
            rearrange(outputs["hidden"].mean(2), "b c d -> b (c d)")
        )
        ce_loss_org = nn.CrossEntropyLoss()(logits_org, label)

        kl_loss = self.args.w_kl * nn.functional.kl_div(
            input=nn.functional.log_softmax(logits_org.detach(), dim=-1),
            target=nn.functional.log_softmax(logits, dim=-1),
            reduction="batchmean",
            log_target=True,
        )

        loss = ce_loss + ce_loss_org + kl_loss

        self.log("train/loss", loss)
        self.log("train/ce_loss", ce_loss)
        self.log("train/ce_loss_org", ce_loss_org)
        self.log("train/kl_loss", kl_loss)
        return loss
