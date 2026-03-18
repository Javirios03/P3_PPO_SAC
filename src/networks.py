import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=512,
        enable_dueling_dqn=False,
        enable_noisy_nets=False,
        enable_distributional=False,
        num_atoms=51,
    ):
        super().__init__()
        self.enable_dueling_dqn = enable_dueling_dqn
        self.enable_noisy_nets = enable_noisy_nets
        self.enable_distributional = enable_distributional
        self.num_atoms = num_atoms
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        Linear = NoisyLinear if enable_noisy_nets else nn.Linear

        if enable_dueling_dqn:
            self.fc_value = Linear(hidden_dim, 256)
            out_v = 1 * (num_atoms if enable_distributional else 1)
            self.value = Linear(256, out_v)

            self.fc_advantage = Linear(hidden_dim, 256)
            out_a = action_dim * (num_atoms if enable_distributional else 1)
            self.advantage = Linear(256, out_a)
        else:
            out_q = action_dim * (num_atoms if enable_distributional else 1)
            self.fc3 = Linear(hidden_dim, out_q)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.enable_dueling_dqn:
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantage(x))
            A = self.advantage(a)

            if self.enable_distributional:
                # reshape: [B, 1*num_atoms] -> [B, 1, num_atoms]
                V = V.view(-1, 1, self.num_atoms)
                # [B, A*num_atoms] -> [B, A, num_atoms]
                A = A.view(-1, self.action_dim, self.num_atoms)
                Q = V + A - A.mean(dim=1, keepdim=True)
                return Q  # logits per atom
            else:
                Q = V + A - A.mean(dim=1, keepdim=True)
                return Q
        else:
            Q = self.fc3(x)
            if self.enable_distributional:
                Q = Q.view(-1, self.action_dim, self.num_atoms)
            return Q


class Pixel_DQN(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        hidden_dim=512,
        enable_dueling_dqn=False,
        enable_noisy_nets=False,
        enable_distributional=False,
        num_atoms=51,
    ):
        super().__init__()
        self.enable_dueling_dqn = enable_dueling_dqn
        self.enable_noisy_nets = enable_noisy_nets
        self.enable_distributional = enable_distributional
        self.num_atoms = num_atoms
        self.action_dim = action_dim

        # Detect input channels from obs shape
        # Grayscale stacked: (stack, H, W)  → in_channels = stack
        # RGB stacked:       (stack, H, W, C) → in_channels = stack * C
        if len(obs_shape) == 3:
            in_channels = obs_shape[0]
            self._needs_permute = False
        else:  # len == 4  →  (stack, H, W, C)
            in_channels = obs_shape[0] * obs_shape[3]
            self._needs_permute = True

        # Convolutional layers for pixel input
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        Linear = NoisyLinear if enable_noisy_nets else nn.Linear

        if enable_dueling_dqn:
            self.fc_value = Linear(7 * 7 * 64, 256)
            out_v = 1 * (num_atoms if enable_distributional else 1)
            self.value = Linear(256, out_v)

            self.fc_advantage = Linear(7 * 7 * 64, 256)
            out_a = action_dim * (num_atoms if enable_distributional else 1)
            self.advantage = Linear(256, out_a)
        else:
            out_q = action_dim * (num_atoms if enable_distributional else 1)
            self.fc3 = Linear(7 * 7 * 64, out_q)

    def forward(self, x):
        # Ensure input is a float tensor on the correct device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(
                x, dtype=torch.float32, device=next(self.parameters()).device
            )

        if self._needs_permute:
            # RGB stacked: [B, Stack, H, W, C] → [B, Stack*C, H, W]
            x = x.permute(0, 1, 4, 2, 3)
            x = x.flatten(start_dim=1, end_dim=2)
        # Grayscale stacked: already [B, Stack, H, W] = [B, C, H, W]

        # Normalize pixel values (0-255 -> 0-1)
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1, end_dim=-1)
        if self.enable_dueling_dqn:
            # Value calc
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantages calc
            a = F.relu(self.fc_advantage(x))
            A = self.advantage(a)

            if self.enable_distributional:
                # reshape: [B, 1*num_atoms] -> [B, 1, num_atoms]
                V = V.view(-1, 1, self.num_atoms)
                # [B, A*num_atoms] -> [B, A, num_atoms]
                A = A.view(-1, self.action_dim, self.num_atoms)
                Q = V + A - A.mean(dim=1, keepdim=True)
                return Q  # logits per atom
            else:
                # Calc Q
                Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.fc3(x)
            if self.enable_distributional:
                Q = Q.view(-1, self.action_dim, self.num_atoms)
        return Q


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.5 / self.in_features**0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(0.5 / self.out_features**0.5)

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
