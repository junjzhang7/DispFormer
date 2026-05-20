import torch
from einops import repeat
from torch import nn



class SufficientIBCompressor(nn.Module):
    def __init__(self, d_model) -> None:
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

    def compress(self, hidden):
        assignment = self.compressor(hidden)  # [bxc, l, d]

        mask = torch.sigmoid(assignment)

        return mask, assignment

    def forward(self, hidden):
        mask, assignment = self.compress(hidden)  # [b, l, 1]

        hidden = mask * hidden
        return hidden


class VTIBCompressor(nn.Module):
    def __init__(self, d_model, dropout=0.1) -> None:
        super().__init__()
        self.local_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )
        self.compressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.init_model()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, hidden):
        l = hidden.size(1)

        # hidden: [bxc, l, d]
        lambda_pos, assignment = self.compress(hidden)  # [bxc, l, 1]
        lambda_neg = 1 - lambda_pos

        static_hidden = hidden.clone().detach()

        # hidden_std: [bxc, d]
        hidden_std, hidden_mean = torch.std_mean(static_hidden, dim=1)
        hidden_mean = repeat(hidden_mean, "b d -> b l d", l=l)  # [bxc, l, d]
        hidden_std = repeat(hidden_std, "b d -> b l d", l=l)  # [bxc, l, d]

        # noisy_hidden_mean: [bxc, l, d]
        noisy_hidden_mean = lambda_pos * hidden + lambda_neg * hidden_mean
        noisy_hidden_std = lambda_neg * hidden_std  # [bxc, l, d]
        # prevent nan
        noisy_hidden_mean = torch.where(
            torch.isfinite(noisy_hidden_mean),
            noisy_hidden_mean,
            torch.zeros_like(noisy_hidden_mean) + 1e-6,
        )
        noisy_hidden_std = torch.where(
            torch.isfinite(noisy_hidden_std),
            noisy_hidden_std,
            torch.zeros_like(noisy_hidden_std) + 1e-6,
        )

        noisy_hidden = (
            noisy_hidden_mean + torch.rand_like(noisy_hidden_mean) * noisy_hidden_std
        )  # [bxc, l, d]

        if self.training:
            p_z_x = torch.distributions.normal.Normal(
                loc=noisy_hidden_mean.sum(1), scale=noisy_hidden_std.sum(1)
            )
            q_z = torch.distributions.normal.Normal(
                loc=hidden_mean.sum(1), scale=hidden_std.sum(1)
            )
            kl = torch.distributions.kl.kl_divergence(p_z_x, q_z)  # [bxc, d]

            kl_loss = kl.sum(dim=-1).mean()
        else:
            kl_loss = 0
        return noisy_hidden, kl_loss

    def compress(self, hidden):
        lacal_feat = self.local_proj(hidden)  # [bxc, l, d//2]
        global_feat = torch.mean(self.global_proj(hidden), dim=1)  # [bxc, 1, d//2]
        global_feat = repeat(global_feat, "... d -> ... l d", l=lacal_feat.size(1))

        p = self.compressor(torch.cat([lacal_feat, global_feat], -1))  # [bxc, l, 1]

        # gumbel sigmoid
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(hidden.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs)

        return gate_inputs, p
