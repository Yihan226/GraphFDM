import torch

import utils
from diffusion import diffusion_utils


class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    def __init__(self, noise_schedule, timesteps):
        super().__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == "custom":
            betas = diffusion_utils.custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer("betas", torch.from_numpy(betas).float())
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=1)
        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]


class MarginalTransition:
    def __init__(self, x_marginals, e_marginals, xe_conditions, ex_conditions, pos_marginals, y_classes, n_nodes):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.pos_classes = len(pos_marginals)
        self.y_classes = y_classes
        self.n_nodes = n_nodes

        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.u_xe = xe_conditions.unsqueeze(0)
        self.u_ex = ex_conditions.unsqueeze(0)
        self.u_pos = pos_marginals.unsqueeze(0).expand(self.pos_classes, -1).unsqueeze(0)

    def get_union_transition(self, n_nodes):
        u_e = self.u_e.repeat(1, n_nodes, n_nodes)
        u_xe = self.u_xe.repeat(1, 1, n_nodes)
        u_ex = self.u_ex.repeat(1, n_nodes, 1)
        top = torch.cat([self.u_x, u_xe], dim=2)
        bottom = torch.cat([u_ex, u_e], dim=2)
        return torch.cat([top, bottom], dim=1)

    def get_Qt(self, beta_t, device, n_nodes=None):
        n_nodes = n_nodes or self.n_nodes
        bs = beta_t.size(0)
        beta_t = beta_t.to(device).view(bs, 1, 1)

        union = self.get_union_transition(n_nodes).to(device)
        union_dim = union.size(-1)
        q_xe = beta_t * union + (1 - beta_t) * torch.eye(union_dim, device=device).unsqueeze(0)

        u_pos = self.u_pos.to(device)
        q_pos = beta_t * u_pos + (1 - beta_t) * torch.eye(self.pos_classes, device=device).unsqueeze(0)

        q_y = None
        if self.y_classes and self.y_classes > 0:
            eye_y = torch.eye(self.y_classes, device=device).unsqueeze(0)
            u_y = torch.ones(1, self.y_classes, self.y_classes, device=device) / self.y_classes
            q_y = beta_t * u_y + (1 - beta_t) * eye_y

        return utils.PlaceHolder(X=q_xe, E=None, y=q_y, pos=q_pos)

    def get_Qt_bar(self, alpha_bar_t, device, n_nodes=None):
        n_nodes = n_nodes or self.n_nodes
        bs = alpha_bar_t.size(0)
        alpha_bar_t = alpha_bar_t.to(device).view(bs, 1, 1)

        union = self.get_union_transition(n_nodes).to(device)
        union_dim = union.size(-1)
        q_xe = alpha_bar_t * torch.eye(union_dim, device=device).unsqueeze(0) + (1 - alpha_bar_t) * union

        u_pos = self.u_pos.to(device)
        q_pos = alpha_bar_t * torch.eye(self.pos_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * u_pos

        q_y = None
        if self.y_classes and self.y_classes > 0:
            eye_y = torch.eye(self.y_classes, device=device).unsqueeze(0)
            u_y = torch.ones(1, self.y_classes, self.y_classes, device=device) / self.y_classes
            q_y = alpha_bar_t * eye_y + (1 - alpha_bar_t) * u_y

        return utils.PlaceHolder(X=q_xe, E=None, y=q_y, pos=q_pos)
