from .layers import *
import math
from model.cnf import CNF
import torch
import torch.nn as nn


class GraphGeo(nn.Module):
    def __init__(self, dim_in, dim_as, dim_z, dim_inner, dim_out, threshold=0.6, lambda_1=0.8, lambda_2=0.4, dropout=0, device="cuda:0"):
        super(GraphGeo, self).__init__()
        self.device = device
        self.threshold = threshold
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # network topology module
        self.dis_co = nn.Linear(dim_in, 3)
        self.kappa = nn.Linear(dim_in, 1)

        # attribute similarity module
        self.n_h = 4
        self.sim_net = Similarity(d_model=dim_in, d_inner=dim_as, n_head=self.n_h,
                                  d_k=dim_as, d_v=dim_as, dropout=0.1)

        # self-supervised regularization
        self.GNNEncoder1 = nn.Sequential(
            nn.Linear(dim_in, dim_as),
            nn.ReLU()
        )
        self.GNNEncoder2 = nn.Linear(dim_as, dim_as)
        self.GNNDecoder = nn.Linear(dim_as, dim_in)

        # feature aggregation
        self.GNN1 = nn.Sequential(
            nn.Linear(dim_in + 2, dim_z),
            nn.ReLU()
        )
        self.GNN2 = nn.Linear(dim_z, dim_z)

        # uncertain-aware inference
        self.mu_net = nn.Linear(dim_z, dim_z)
        self.var_net = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.Softplus()
        )
        self.adj_rec = nn.Linear(2 * dim_z, 1)

        self.cnf = CNF(dim_z, dim_z, dim_z)

        self.pred = nn.Sequential(
            nn.Linear(dim_z * 2, dim_inner),
            nn.ReLU(),
            nn.Linear(dim_inner, dim_out)
        )

    def forward(self, lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay):
        X = torch.cat((lm_X, tg_X), dim=0)
        log_delays = torch.cat((lm_delay, tg_delay), dim=0)
        """ IP Host Connecting """
        # network topology
        coe = self.dis_co(X)
        alpha, beta, gamma = coe[:, 0], coe[:, 1], coe[:, 2]
        distance = alpha * log_delays + beta * torch.ones_like(log_delays, device=self.device) + gamma
        edge_nt = torch.abs(distance.repeat(distance.shape[0], 1) - distance.repeat(distance.shape[0], 1).T)
        edge_nt = torch.exp(-edge_nt) * torch.relu(self.kappa(X))

        # attribute similarity
        _, edge_as = self.sim_net(X.unsqueeze(dim=0), X.unsqueeze(dim=0))
        edge_as = (edge_as.sum(dim=1) / self.n_h).squeeze(dim=0)
        adj = (edge_nt + edge_as) * gen_mask(len(lm_X), len(tg_X)).to(self.device)
        topk, indices = torch.topk(adj, math.ceil(len(lm_X) * self.threshold), dim=1)
        adj = torch.zeros_like(adj, device=self.device).scatter_(1, indices, topk)
        adj_max = torch.max(adj, dim=1, keepdim=True)[0]
        adj_exp = torch.exp(adj - adj_max)
        adj_exp = adj_exp * (adj != 0).type(torch.FloatTensor).to(self.device)
        adj_soft = adj_exp / torch.sum(adj_exp, dim=1, keepdim=True)
        adj = adj_soft

        # self-supervised reg
        h_enc = self.GNNEncoder2(torch.mm(adj, self.GNNEncoder1(torch.mm(adj, X))))
        x_dec = self.GNNDecoder(h_enc)
        g_loss = nn.MSELoss()(x_dec, X)

        """ IP Knowledge Aggregation with Uncertainty """
        # uncertainty-aware gnn
        Y = torch.cat((lm_Y, torch.zeros_like(tg_Y)), dim=0)
        input = torch.cat((X, Y), dim=1)
        h_enc_u = self.GNN2(torch.mm(adj, self.GNN1(torch.mm(adj, input))))
        z_mu, z_sigma = self.mu_net(h_enc_u), self.var_net(h_enc_u) + 1e-10
        z_0 = torch.distributions.Normal(loc=z_mu, scale=z_sigma).rsample()
        z_t, logp_diff_t = self.cnf(z_0)

        Z_ij = torch.cat((z_t.repeat(z_t.shape[0], 1, 1), z_t.repeat(z_t.shape[0], 1, 1).permute(1, 0, 2)), dim=-1)
        rec_loss = ((adj != 0) * torch.log(1 + torch.exp(self.adj_rec(Z_ij))).squeeze(dim=-1)).mean()
        elbo = rec_loss  + (-0.5 * (z_t * z_t).mean()) - (-1) * (0.5 * (z_0 * z_0).mean() + logp_diff_t.view(-1).mean())

        y_pred = self.pred(torch.cat((h_enc_u, z_t), dim=1)) + lm_Y.mean(dim=0)

        return y_pred[-len(tg_X):], g_loss * self.lambda_1 - elbo * self.lambda_2


def gen_mask(num_lm, num_tg):
    mask = torch.ones((num_lm + num_tg, num_lm + num_tg), device="cuda:0")
    mask[-num_tg:, -num_tg:] = torch.eye(num_tg, device="cuda:0")
    return mask
