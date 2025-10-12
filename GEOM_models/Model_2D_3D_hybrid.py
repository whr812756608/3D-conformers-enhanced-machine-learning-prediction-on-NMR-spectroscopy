# Model_3D_NMR.py
"""
3D-Enhanced NMR Shift Prediction Model with adaptive 2D/3D mixing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set, NNConv

class nmrMPNN(nn.Module):
    """
    Pure 2D MPNN baseline (your implementation).
    Uses NNConv + GRU + Set2Set. No .pos or 3D features are used.
    Forward signature expects (g, n_nodes_per_graph, masks).
    """

    def __init__(self, node_in_feats, edge_in_feats,
                 node_feats=64, embed_feats=256,
                 num_step_message_passing=5,
                 num_step_set2set=3, num_layer_set2set=1,
                 hidden_feats=512, prob_dropout=0.1):
        super(nmrMPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, node_feats), nn.Tanh()
        )

        self.num_step_message_passing = num_step_message_passing
        self.node_feats = node_feats

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, node_feats * node_feats), nn.ReLU()
        )

        self.gnn_layer = NNConv(
            in_channels=node_feats,
            out_channels=node_feats,
            nn=edge_network,
            aggr='add'
        )

        self.gru = nn.GRU(node_feats, node_feats)

        # Input channels: node_feats * (1 + num_step_message_passing)
        node_aggr_dim = node_feats * (1 + num_step_message_passing)

        self.readout = Set2Set(
            in_channels=node_aggr_dim,
            processing_steps=num_step_set2set
        )

        # Set2Set outputs 2x the input dimension
        graph_dim = node_aggr_dim * 2

        # Total input: node features + graph features
        predict_input_dim = node_aggr_dim + graph_dim

        self.predict = nn.Sequential(
            nn.Linear(predict_input_dim, hidden_feats), nn.PReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, 1)
        )

    def forward(self, g, n_nodes, masks):
        # Project atoms
        node_feats = self.project_node_feats(g.x)

        edge_index = g.edge_index
        edge_feats = g.edge_attr

        # Aggregate over message-passing steps (with GRU updates)
        node_aggr = [node_feats]
        for _ in range(self.num_step_message_passing):
            msg = self.gnn_layer(node_feats, edge_index, edge_feats).unsqueeze(0)
            _, node_feats = self.gru(msg, node_feats.unsqueeze(0))
            node_feats = node_feats.squeeze(0)
            node_aggr.append(node_feats)

        node_aggr = torch.cat(node_aggr, dim=1)  # [N, node_aggr_dim]

        # Graph-level Set2Set, then broadcast back to nodes
        graph_embed_feats = self.readout(node_aggr, g.batch)              # [B, 2*node_aggr_dim]
        graph_embed_feats = torch.repeat_interleave(graph_embed_feats, n_nodes, dim=0)  # [N, 2*node_aggr_dim]

        out = self.predict(torch.hstack([node_aggr, graph_embed_feats]))  # [N, 1]
        return out.flatten()[masks]


class NMR2DMPNN(nn.Module):
    """
    Thin wrapper so the 2D baseline can be called as: pred = model(npz_data) -> pred[npz_data.mask]
    """
    def __init__(self,
                 node_in_dim: int = 74,
                 edge_in_dim: int = 9,
                 node_feats: int = 64,
                 embed_feats: int = 256,
                 num_step_message_passing: int = 5,
                 num_step_set2set: int = 3,
                 hidden_feats: int = 512,
                 prob_dropout: float = 0.1):
        super().__init__()
        self.core = nmrMPNN(
            node_in_feats=node_in_dim,
            edge_in_feats=edge_in_dim,
            node_feats=node_feats,
            embed_feats=embed_feats,
            num_step_message_passing=num_step_message_passing,
            num_step_set2set=num_step_set2set,
            hidden_feats=hidden_feats,
            prob_dropout=prob_dropout,
        )

    def forward(self, data):
        n_per_graph = torch.bincount(
            data.batch, minlength=int(data.batch.max().item()) + 1
        )
        return self.core(data, n_per_graph, data.mask)



# ===================== 3D Feature Encoders =====================

class GaussianBasis(nn.Module):
    """SchNet-style RBF distance encoding"""

    def __init__(self, start=0.0, stop=8.0, num_gaussians=32):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / ((offset[1] - offset[0]) ** 2)

    def forward(self, dist):
        """dist: [E] -> [E, num_gaussians]"""
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * dist ** 2)

# ===================== ComENet-lite Angular Features =====================

class AngleFourier(nn.Module):
    """
    Fourier features for angles θ in [0, π]. Returns:
      [cosθ, sinθ, cos(1θ), sin(1θ), ..., cos(Kθ), sin(Kθ)]  (size = 2K + 2)
    """
    def __init__(self, K: int = 4):
        super().__init__()
        self.K = K

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        # angles: [M] (radians, in [0, π])
        cos1 = torch.cos(angles)
        sin1 = torch.sin(angles)
        feats = [cos1, sin1]
        for k in range(1, self.K + 1):
            feats += [torch.cos(k * angles), torch.sin(k * angles)]
        return torch.stack(feats, dim=-1)  # [M, 2K+2]


@torch.no_grad()
def compute_node_angle_features(data, K: int = 4, eps: float = 1e-8):
    """
    For each node i, compute angles between all incident bond vectors (j->i) and (k->i).
    Aggregate Fourier-encoded angles to get a per-node "ComENet-lite" angular embedding.

    Returns: node_angle_feats [N, 2K+2], zero for degree<2 nodes.
    """
    device = data.x.device
    N = data.x.size(0)
    src, dst = data.edge_index  # edges are directed; treat as incoming to 'dst'
    pos = data.pos

    # incoming vectors v_{j->i} = pos[j] - pos[i] for all edges j->i
    v = pos[src] - pos[dst]                            # [E, 3]
    # group edges by center node i = dst
    # we need, for each i, all pairwise angles between v’s in its group

    # sort by dst to build segments
    order = torch.argsort(dst)
    dst_sorted = dst[order]
    v_sorted = v[order]

    # segment boundaries per node
    # idx where dst changes
    changes = torch.nonzero(torch.diff(dst_sorted, prepend=dst_sorted[:1]-1), as_tuple=False).squeeze(-1)
    # counts per node
    counts = torch.bincount(dst, minlength=N)  # [N]

    # pre-allocate node features
    fourier = AngleFourier(K).to(device)
    out = torch.zeros(N, 2*K + 2, device=device)

    start = 0
    for i in range(N):
        deg = counts[i].item()
        if deg < 2:
            continue
        end = start + deg
        vi = v_sorted[start:end]                 # [deg, 3]

        # pairwise angles among vi rows
        # normalize
        nrm = torch.clamp(vi.norm(dim=-1, keepdim=True), min=eps)
        u = vi / nrm                             # [deg, 3]
        # cosine matrix U U^T in [-1,1]
        cosmat = (u @ u.t()).clamp(-1.0, 1.0)    # [deg, deg]
        # take upper triangle (j<k) to avoid duplicates
        iu, ju = torch.triu_indices(deg, deg, offset=1, device=device)
        cos_vals = cosmat[iu, ju]                # [M]
        angles = torch.acos(cos_vals)            # [M] in [0, π]

        ang_enc = fourier(angles)                # [M, 2K+2]
        # aggregate (mean) => per-node embedding
        out[i] = ang_enc.mean(dim=0)

        start = end

    return out  # [N, 2K+2]


# ===================== Message Passing Layers =====================

class SchNetInteraction(nn.Module):
    """SchNet continuous-filter convolution"""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_dim = node_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_features):
        src, dst = edge_index
        edge_weights = self.edge_mlp(edge_features)
        messages = x[src] * edge_weights

        out = torch.zeros_like(x)
        out.index_add_(0, dst, messages)
        out = self.node_mlp(out)
        return x + out


class EGNNLayer(nn.Module):
    """E(n) Equivariant Graph Neural Network layer"""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, pos, edge_index, edge_features):
        src, dst = edge_index

        rel_pos = pos[src] - pos[dst]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)

        edge_input = torch.cat([x[src], x[dst], edge_features, dist], dim=-1)
        edge_messages = self.edge_mlp(edge_input)

        node_messages = torch.zeros(x.size(0), edge_messages.size(1), device=x.device)
        node_messages.index_add_(0, dst, edge_messages)

        node_input = torch.cat([x, node_messages], dim=-1)
        x_update = self.node_mlp(node_input)

        return x + x_update


# ===================== Main Models =====================

class NMR3DNet(nn.Module):
    """
    2D/3D NMR Shift Prediction Network built around a 2D MPNN backbone.
    - 2D path: NNConv + GRU + (per-step concat) -> Set2Set -> predictor
    - 3D path: adds SchNet/EGNN on top of the *2D hidden states*
               using RBF(distance)-augmented edge features, then fuses 2D/3D.
    """

    def __init__(
        self,
        # ---- fixed input dims
        node_in_dim: int = 74,
        edge_in_dim: int = 9,

        # ---- 2D MPNN backbone (your previous 2D model) ----
        node_hidden: int = 128,          # hidden width for nodes (also used by 3D heads)
        mpnn_embed: int = 256,           # edge/node MLP size in the 2D backbone
        num_step_message_passing: int = 5,
        set2set_steps: int = 3,

        # ---- 3D encoders (SchNet/EGNN) ----
        edge_hidden: int = 256,          # edge MLP size for SchNet/EGNN
        n_interactions: int = 4,
        n_rbf: int = 32,
        rbf_cutoff: float = 8.0,
        use_3d: bool = True,
        use_schnet: bool = True,
        use_egnn: bool = True,
        use_comenet: bool = True,
        comenet_K: int = 4,
        comenet_into_2d: bool = False,
        readout_hidden: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.use_3d = use_3d
        self.use_schnet = use_schnet
        self.use_egnn = use_egnn
        self.use_comenet = use_comenet
        self.n_interactions = n_interactions
        self.comenet_K = comenet_K
        self.comenet_into_2d = comenet_into_2d

        # ===================== 2D BACKBONE (NNConv + GRU) =====================
        # project nodes into backbone hidden width
        self.node_proj_2d = nn.Sequential(
            nn.Linear(node_in_dim, mpnn_embed), nn.ReLU(),
            nn.Linear(mpnn_embed, mpnn_embed), nn.ReLU(),
            nn.Linear(mpnn_embed, mpnn_embed), nn.ReLU(),
            nn.Linear(mpnn_embed, node_hidden), nn.Tanh()
        )

        # edge MLP -> NNConv kernel (2D only; no 3D features here)
        self.edge_mlp_2d = nn.Sequential(
            nn.Linear(edge_in_dim, mpnn_embed), nn.ReLU(),
            nn.Linear(mpnn_embed, mpnn_embed), nn.ReLU(),
            nn.Linear(mpnn_embed, mpnn_embed), nn.ReLU(),
            nn.Linear(mpnn_embed, node_hidden * node_hidden), nn.ReLU()
        )
        self.nnconv_2d = NNConv(
            in_channels=node_hidden,
            out_channels=node_hidden,
            nn=self.edge_mlp_2d,
            aggr='add'
        )
        self.gru = nn.GRU(node_hidden, node_hidden)

        # ===================== 3D AUGMENTATION (on top of 2D states) =====================
        # project raw edge_attr for SchNet/EGNN; then concat RBF(dist)
        self.edge_proj_3d = nn.Linear(edge_in_dim, edge_hidden)
        if self.use_3d:
            self.rbf_encoder = GaussianBasis(0.0, rbf_cutoff, n_rbf)
            edge_dim_with_3d = edge_hidden + n_rbf
        else:
            edge_dim_with_3d = edge_hidden  # unused if not 3D

        if self.use_schnet:
            self.schnet_layers = nn.ModuleList([
                SchNetInteraction(node_hidden, edge_dim_with_3d, edge_hidden)
                for _ in range(n_interactions)
            ])

        if self.use_egnn and self.use_3d:
            self.egnn_layers = nn.ModuleList([
                EGNNLayer(node_hidden, edge_dim_with_3d, edge_hidden)
                for _ in range(n_interactions)
            ])

        # fuse SchNet vs EGNN (3D heads)
        if self.use_schnet and self.use_egnn and self.use_3d:
            self.fusion_gate_3d = nn.Sequential(
                nn.Linear(node_hidden * 2, node_hidden),
                nn.Sigmoid()
            )

        # fuse final 2D backbone vs 3D head
        if self.use_3d:
            self.fusion_gate_2d3d = nn.Sequential(
                nn.Linear(node_hidden * 2, node_hidden),
                nn.Sigmoid()
            )

        if self.use_comenet:
            # per-node angle feats have dim (2K+2); project to node_hidden
            self.comenet_node_proj = nn.Sequential(
                nn.Linear(2*comenet_K + 2, node_hidden),
                nn.SiLU(),
                nn.Linear(node_hidden, node_hidden)
            )

        # ===================== READOUT / PREDICTOR =====================
        # We concatenate all per-step states (like your 2D model)
        self.node_aggr_dim = node_hidden * (1 + num_step_message_passing)
        self.set2set = Set2Set(self.node_aggr_dim, processing_steps=set2set_steps)

        self.predictor = nn.Sequential(
            nn.Linear(self.node_aggr_dim + 2 * self.node_aggr_dim, readout_hidden),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden, readout_hidden),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden, 1)
        )

        # store for forward loop
        self.num_step_message_passing = num_step_message_passing
        self.node_hidden = node_hidden

    def _build_edge_features_3d(self, data, edge_features_lin):
        """edge_features_lin: projected edge features (edge_hidden), add RBF(dist)"""
        src, dst = data.edge_index
        distances = torch.norm(data.pos[src] - data.pos[dst], dim=-1)  # [E]
        rbf = self.rbf_encoder(distances)                              # [E, n_rbf]
        return torch.cat([edge_features_lin, rbf], dim=-1)             # [E, edge_hidden+n_rbf]

    def forward(self, data):
        """
        npz_data: PyG Data with x, edge_index, edge_attr, pos(optional), mask, batch
        """
        # ========== 2D BACKBONE ==========
        h = self.node_proj_2d(data.x)  # [N, F]
        H_steps = [h]

        for _ in range(self.num_step_message_passing):
            m2d = self.nnconv_2d(h, data.edge_index, data.edge_attr)  # [N, F]
            # GRU update
            mseq = m2d.unsqueeze(0)
            hseq = h.unsqueeze(0)
            _, h_next = self.gru(mseq, hseq)
            h = h_next.squeeze(0)
            H_steps.append(h)

        # Concatenate per-step features (like your 2D nmrMPNN)
        H2d = torch.cat(H_steps, dim=1)  # [N, F*(1+T)]

        # If 3D disabled: readout directly from 2D backbone (pure 2D model)
        if not self.use_3d or not hasattr(data, 'pos'):
            G = self.set2set(H2d, data.batch)  # [B, 2*F*(1+T)]
            Gb = G[data.batch]  # [N, 2*F*(1+T)]
            y = self.predictor(torch.hstack([H2d, Gb])).squeeze(-1)
            return y[data.mask]

        # ========== 3D AUGMENTATION ON TOP OF 2D HIDDEN ==========
        # Project edges, add RBF(dist)
        edge_lin = self.edge_proj_3d(data.edge_attr)  # [E, edge_hidden]
        edge_feat_3d = self._build_edge_features_3d(data, edge_lin)  # [E, edge_hidden+n_rbf]

        # --- ComENet-lite per-node angular embedding (compute BEFORE heads) ---
        angle_embed = None
        if getattr(self, "use_comenet", False):
            angle_embed = compute_node_angle_features(data, K=self.comenet_K)  # [N, 2K+2]
            angle_embed = self.comenet_node_proj(angle_embed)  # [N, F]
            # If requested, inject into the current 2D state 'h'
            if getattr(self, "comenet_into_2d", False):
                h = h + angle_embed

        # Run SchNet/EGNN starting from the final 2D state h
        x_schnet = x_egnn = None

        if self.use_schnet:
            x_schnet = h
            for i in range(self.n_interactions):
                x_schnet = self.schnet_layers[i](x_schnet, data.edge_index, edge_feat_3d)
            if angle_embed is not None:
                x_schnet = x_schnet + angle_embed

        if self.use_egnn:
            x_egnn = h
            for i in range(self.n_interactions):
                x_egnn = self.egnn_layers[i](x_egnn, data.pos, data.edge_index, edge_feat_3d)
            if angle_embed is not None:
                x_egnn = x_egnn + angle_embed

        # Fuse 3D heads robustly
        if (x_schnet is not None) and (x_egnn is not None):
            gate3d = self.fusion_gate_3d(torch.cat([x_schnet, x_egnn], dim=-1))  # [N, F]
            x3d = gate3d * x_schnet + (1.0 - gate3d) * x_egnn
        elif x_schnet is not None:
            x3d = x_schnet
        else:
            x3d = x_egnn  # must not be None because 3D is enabled

        # Fuse 2D backbone vs 3D augmentation (per-atom gate)
        gate2d3d = self.fusion_gate_2d3d(torch.cat([h, x3d], dim=-1))  # [N, F]
        h_fused = gate2d3d * x3d + (1.0 - gate2d3d) * h  # [N, F]

        # Replace the last block in H2d with the fused h_fused (keeps “per-step” shape)
        F = self.node_hidden
        Hcat = torch.cat([H2d[:, :-F], h_fused], dim=1)  # [N, F*(1+T)]

        # Readout + predict
        G = self.set2set(Hcat, data.batch)  # [B, 2*F*(1+T)]
        Gb = G[data.batch]  # [N, 2*F*(1+T)]
        y = self.predictor(torch.hstack([Hcat, Gb])).squeeze(-1)  # [N]
        return y[data.mask]


# class AdaptiveNMR3DNet(nn.Module):
#     """
#     Adaptive hybrid model that learns when to use 3D features
#     Automatically switches between 2D and 3D based on atom features
#     """
#
#     def __init__(
#             self,
#             node_in_dim=74,
#             edge_in_dim=9,
#             node_hidden=128,
#             edge_hidden=256,
#             n_interactions=4,
#             n_rbf=32,
#             rbf_cutoff=8.0,
#             readout_hidden=512,
#             dropout=0.1,
#             selector_hidden=64,
#             use_adaptive=True,  # NEW: Enable/disable adaptive mixing
#     ):
#         super().__init__()
#
#         self.use_adaptive = use_adaptive
#         self.node_hidden = node_hidden
#
#         # Shared components
#         self.node_proj = nn.Linear(node_in_dim, node_hidden)
#         self.edge_proj = nn.Linear(edge_in_dim, edge_hidden)
#         self.rbf_encoder = GaussianBasis(0, rbf_cutoff, n_rbf)
#
#         # 2D branch (SchNet only, no 3D features)
#         self.schnet_2d_layers = nn.ModuleList([
#             SchNetInteraction(node_hidden, edge_hidden, edge_hidden)
#             for _ in range(n_interactions)
#         ])
#
#         # 3D branch (SchNet + EGNN with 3D features)
#         self.schnet_3d_layers = nn.ModuleList([
#             SchNetInteraction(node_hidden, edge_hidden + n_rbf, edge_hidden)
#             for _ in range(n_interactions)
#         ])
#
#         self.egnn_layers = nn.ModuleList([
#             EGNNLayer(node_hidden, edge_hidden + n_rbf, edge_hidden)
#             for _ in range(n_interactions)
#         ])
#
#         # 3D branch fusion
#         self.fusion_3d = nn.Sequential(
#             nn.Linear(node_hidden * 2, node_hidden),
#             nn.Sigmoid()
#         )
#
#         # Adaptive selector network
#         if self.use_adaptive:
#             self.selector = nn.Sequential(
#                 nn.Linear(node_in_dim, selector_hidden),  # Use raw features
#                 nn.ReLU(),
#                 nn.Linear(selector_hidden, selector_hidden),
#                 nn.ReLU(),
#                 nn.Linear(selector_hidden, 1),
#                 nn.Sigmoid()  # 0 = use 2D, 1 = use 3D
#             )
#
#         with torch.no_grad():
#             self.selector[-2].bias.fill_(2.0)
#         # Shared readout and predictor
#         self.set2set = Set2Set(node_hidden, processing_steps=3)
#
#         self.predictor = nn.Sequential(
#             nn.Linear(node_hidden + node_hidden * 2, readout_hidden),
#             nn.PReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(readout_hidden, readout_hidden),
#             nn.PReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(readout_hidden, 1)
#         )
#
#     def forward(self, npz_data):
#         """
#         npz_data: PyG Data with x, edge_index, edge_attr, pos, mask, batch
#         """
#         # Project inputs
#         x_init = self.node_proj(npz_data.x)
#         edge_features = self.edge_proj(npz_data.edge_attr)
#
#         # === 2D Branch (no 3D features) ===
#         x_2d = x_init
#         for layer in self.schnet_2d_layers:
#             x_2d = layer(x_2d, npz_data.edge_index, edge_features)
#
#         # === 3D Branch (with 3D features) ===
#         if hasattr(npz_data, 'pos'):
#             # Add RBF distance features
#             src, dst = npz_data.edge_index
#             distances = torch.norm(npz_data.pos[src] - npz_data.pos[dst], dim=-1)
#             rbf_features = self.rbf_encoder(distances)
#             edge_features_3d = torch.cat([edge_features, rbf_features], dim=-1)
#
#             # SchNet with 3D
#             x_schnet_3d = x_init
#             for layer in self.schnet_3d_layers:
#                 x_schnet_3d = layer(x_schnet_3d, npz_data.edge_index, edge_features_3d)
#
#             # EGNN
#             x_egnn = x_init
#             for layer in self.egnn_layers:
#                 x_egnn = layer(x_egnn, npz_data.pos, npz_data.edge_index, edge_features_3d)
#
#             # Fuse 3D components
#             gate_3d = self.fusion_3d(torch.cat([x_schnet_3d, x_egnn], dim=-1))
#             x_3d = gate_3d * x_schnet_3d + (1 - gate_3d) * x_egnn
#         else:
#             # No coordinates available, fall back to 2D
#             x_3d = x_2d
#
#         # === Adaptive Mixing ===
#         if self.use_adaptive and hasattr(npz_data, 'pos'):
#             # Learn per-atom mixing weights based on atom features
#             mixing_weights = self.selector(npz_data.x)  # [N, 1]
#             x_out = mixing_weights * x_3d + (1 - mixing_weights) * x_2d
#         else:
#             # If adaptive disabled, use 3D only (or 2D if no pos)
#             x_out = x_3d if hasattr(npz_data, 'pos') else x_2d
#
#         # Graph-level readout
#         graph_features = self.set2set(x_out, npz_data.batch)
#         graph_features_per_atom = graph_features[npz_data.batch]
#
#         # Prediction
#         combined = torch.cat([x_out, graph_features_per_atom], dim=-1)
#         predictions = self.predictor(combined).squeeze(-1)
#         predictions = predictions[npz_data.mask]
#
#         return predictions
#
#     def get_mixing_weights(self, npz_data):
#         """Get the learned 2D/3D mixing weights for analysis"""
#         if not self.use_adaptive:
#             return None
#         with torch.no_grad():
#             weights = self.selector(npz_data.x)
#             return weights[npz_data.mask].cpu().numpy()


# ===================== Training Utilities =====================

def train_epoch(model, loader, optimizer, device, y_mean, y_std):
    """Single training epoch"""
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        target = data.y[data.mask]

        pred_norm = (pred - y_mean) / y_std
        target_norm = (target - y_mean) / y_std

        loss = F.l1_loss(pred_norm, target_norm)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_targets = []

    for data in loader:
        data = data.to(device)
        pred = model(data)
        target = data.y[data.mask]

        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


@torch.no_grad()
def check_3d_features(model, loader, device):
    """Debug: Check if 3D features are being used properly"""
    model.eval()

    for data in loader:
        data = data.to(device)
        src, dst = data.edge_index

        if hasattr(data, 'pos'):
            distances = torch.norm(data.pos[src] - data.pos[dst], dim=-1)
            print(f"\n3D Feature Diagnostics:")
            print(
                f"  Bond distances: min={distances.min():.3f}, max={distances.max():.3f}, mean={distances.mean():.3f} Å")

            # Check mixing weights for adaptive model
            if hasattr(model, 'get_mixing_weights'):
                weights = model.get_mixing_weights(data)
                if weights is not None:
                    print(
                        f"  Mixing weights (3D usage): min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
                    print(f"  Atoms preferring 2D (<0.3): {(weights < 0.3).sum()}")
                    print(f"  Atoms preferring 3D (>0.7): {(weights > 0.7).sum()}")

            short = (distances < 0.8).sum().item()
            long = (distances > 2.5).sum().item()
            print(f"  Suspicious bonds: {short} too short (<0.8Å), {long} too long (>2.5Å)")

        break