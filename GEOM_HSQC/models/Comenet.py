from torch_cluster import radius_graph
from torch_geometric.nn import GraphConv, GraphNorm
from torch_geometric.nn import inits
from torch_geometric.nn import global_mean_pool

# from .features import angle_emb, torsion_emb

from torch_scatter import scatter, scatter_min

from torch.nn import Embedding

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import math
from math import sqrt

try:
    import sympy as sym
except ImportError:
    sym = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import math
import torch
import sympy as sym
import numpy as np
from scipy.optimize import brentq
from scipy import special as sp

from scipy.special import binom
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.typing import OptPairTensor, Adj, OptTensor


def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return sp.spherical_jn(n, r)


def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols("x")
    # j_i = (-x)^i * (1/x * d/dx)^î * sin(x)/x
    j = [sym.sin(x) / x]  # j_0
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        j += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return j


def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    Returns:
        bess_basis: list
            Bessel basis formulas taking in a single argument x.
            Has length n where each element has length k. -> In total n*k many.
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = (
            1 / np.array(normalizer_tmp) ** 0.5
        )  # sqrt(2/(j_l+1)**2) , sqrt(1/c**3) not taken into account yet
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(
                    normalizer[order][i] * f[order].subs(x, zeros[order, i] * x)
                )
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l, m):
    """Computes the constant pre-factor for the spherical harmonic of degree l and order m.
    Parameters
    ----------
        l: int
            Degree of the spherical harmonic. l >= 0
        m: int
            Order of the spherical harmonic. -l <= m <= l
    Returns
    -------
        factor: float
    """
    # sqrt((2*l+1)/4*pi * (l-m)!/(l+m)! )
    return (
        (2 * l + 1)
        / (4 * np.pi)
        * np.math.factorial(l - abs(m))
        / np.math.factorial(l + abs(m))
    ) ** 0.5


def associated_legendre_polynomials(L, zero_m_only=True, pos_m_only=True):
    """Computes string formulas of the associated legendre polynomials up to degree L (excluded).
    Parameters
    ----------
        L: int
            Degree up to which to calculate the associated legendre polynomials (degree L is excluded).
        zero_m_only: bool
            If True only calculate the polynomials for the polynomials where m=0.
        pos_m_only: bool
            If True only calculate the polynomials for the polynomials where m>=0. Overwritten by zero_m_only.
    Returns
    -------
        polynomials: list
            Contains the sympy functions of the polynomials (in total L many if zero_m_only is True else L^2 many).
    """
    # calculations from http://web.cmb.usc.edu/people/alber/Software/tomominer/docs/cpp/group__legendre__polynomials.html
    z = sym.symbols("z")
    P_l_m = [[0] * (2 * l + 1) for l in range(L)]  # for order l: -l <= m <= l

    P_l_m[0][0] = 1
    if L > 0:
        if zero_m_only:
            # m = 0
            P_l_m[1][0] = z
            for l in range(2, L):
                P_l_m[l][0] = sym.simplify(
                    ((2 * l - 1) * z * P_l_m[l - 1][0] - (l - 1) * P_l_m[l - 2][0]) / l
                )
            return P_l_m
        else:
            # for m >= 0
            for l in range(1, L):
                P_l_m[l][l] = sym.simplify(
                    (1 - 2 * l) * (1 - z ** 2) ** 0.5 * P_l_m[l - 1][l - 1]
                )  # P_00, P_11, P_22, P_33

            for m in range(0, L - 1):
                P_l_m[m + 1][m] = sym.simplify(
                    (2 * m + 1) * z * P_l_m[m][m]
                )  # P_10, P_21, P_32, P_43

            for l in range(2, L):
                for m in range(l - 1):  # P_20, P_30, P_31
                    P_l_m[l][m] = sym.simplify(
                        (
                            (2 * l - 1) * z * P_l_m[l - 1][m]
                            - (l + m - 1) * P_l_m[l - 2][m]
                        )
                        / (l - m)
                    )

            if not pos_m_only:
                # for m < 0: P_l(-m) = (-1)^m * (l-m)!/(l+m)! * P_lm
                for l in range(1, L):
                    for m in range(1, l + 1):  # P_1(-1), P_2(-1) P_2(-2)
                        P_l_m[l][-m] = sym.simplify(
                            (-1) ** m
                            * np.math.factorial(l - m)
                            / np.math.factorial(l + m)
                            * P_l_m[l][m]
                        )

            return P_l_m


def real_sph_harm(L, spherical_coordinates, zero_m_only=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
    Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.
    Parameters
    ----------
        L: int
            Degree up to which to calculate the spherical harmonics (degree L is excluded).
        spherical_coordinates: bool
            - True: Expects the input of the formula strings to be phi and theta.
            - False: Expects the input of the formula strings to be x, y and z.
        zero_m_only: bool
            If True only calculate the harmonics where m=0.
    Returns
    -------
        Y_lm_real: list
            Computes formula strings of the the real part of the spherical harmonics up
            to degree L (where degree L is not excluded).
            In total L^2 many sph harm exist up to degree L (excluded). However, if zero_m_only only is True then
            the total count is reduced to be only L many.
    """
    z = sym.symbols("z")
    P_l_m = associated_legendre_polynomials(L, zero_m_only)
    if zero_m_only:
        # for all m != 0: Y_lm = 0
        Y_l_m = [[0] for l in range(L)]
    else:
        Y_l_m = [[0] * (2 * l + 1) for l in range(L)]  # for order l: -l <= m <= l

    # convert expressions to spherical coordiantes
    if spherical_coordinates:
        # replace z by cos(theta)
        theta = sym.symbols("theta")
        for l in range(L):
            for m in range(len(P_l_m[l])):
                if not isinstance(P_l_m[l][m], int):
                    P_l_m[l][m] = P_l_m[l][m].subs(z, sym.cos(theta))

    ## calculate Y_lm
    # Y_lm = N * P_lm(cos(theta)) * exp(i*m*phi)
    #             { sqrt(2) * (-1)^m * N * P_l|m| * sin(|m|*phi)   if m < 0
    # Y_lm_real = { Y_lm                                           if m = 0
    #             { sqrt(2) * (-1)^m * N * P_lm * cos(m*phi)       if m > 0

    for l in range(L):
        Y_l_m[l][0] = sym.simplify(sph_harm_prefactor(l, 0) * P_l_m[l][0])  # Y_l0

    if not zero_m_only:
        phi = sym.symbols("phi")
        for l in range(1, L):
            # m > 0
            for m in range(1, l + 1):
                Y_l_m[l][m] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m
                    * sph_harm_prefactor(l, m)
                    * P_l_m[l][m]
                    * sym.cos(m * phi)
                )
            # m < 0
            for m in range(1, l + 1):
                Y_l_m[l][-m] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m
                    * sph_harm_prefactor(l, -m)
                    * P_l_m[l][m]
                    * sym.sin(m * phi)
                )

        # convert expressions to cartesian coordinates
        if not spherical_coordinates:
            # replace phi by atan2(y,x)
            x = sym.symbols("x")
            y = sym.symbols("y")
            for l in range(L):
                for m in range(len(Y_l_m[l])):
                    Y_l_m[l][m] = sym.simplify(Y_l_m[l][m].subs(phi, sym.atan2(y, x)))
    return Y_l_m


class angle_emb(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0):
        super(angle_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff

        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=True
        )
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        m = 0
        for l in range(len(Y_lm)):
            if l == 0:
                first_sph = sym.lambdify([theta], Y_lm[l][m], modules)
                self.sph_funcs.append(
                    lambda theta: torch.zeros_like(theta) + first_sph(theta)
                )
            else:
                self.sph_funcs.append(sym.lambdify([theta], Y_lm[l][m], modules))
            for n in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][n], modules)
                )

    def forward(self, dist, angle):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        sbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)
        n, k = self.num_spherical, self.num_radial
        out = (rbf.view(-1, n, k) * sbf.view(-1, n, 1)).view(-1, n * k)
        return out


class torsion_emb(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0):
        super(torsion_emb, self).__init__()
        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff

        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=False
        )
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        phi = sym.symbols("phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        for l in range(len(Y_lm)):
            for m in range(len(Y_lm[l])):
                if (
                        l == 0
                ):
                    first_sph = sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    self.sph_funcs.append(
                        lambda theta, phi: torch.zeros_like(theta)
                                           + first_sph(theta, phi)
                    )
                else:
                    self.sph_funcs.append(
                        sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    )
            for j in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][j], modules)
                )

        self.register_buffer(
            "degreeInOrder", torch.arange(num_spherical) * 2 + 1, persistent=False
        )

    def forward(self, dist, theta, phi):
        dist = dist / self.cutoff
        # assert not (dist==0).any(), 'zeros in dist'
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        sbf = torch.stack([f(theta, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        rbf = rbf.view((-1, n, k)).repeat_interleave(self.degreeInOrder, dim=1).view((-1, n ** 2 * k))
        sbf = sbf.repeat_interleave(k, dim=1)
        out = rbf * sbf
        # assert not torch.isnan(dist).any(), "NaN values found in dist after division"
        # assert not torch.isnan(rbf).any(), "NaN values found in rbf"
        # assert not torch.isnan(sbf).any(), "NaN values found in sbf"


        return out


def swish(x):
    return x * torch.sigmoid(x)

class Linear(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 weight_initializer='glorot',
                 bias_initializer='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == 'glorot':
                inits.glorot(self.weight)
            elif self.weight_initializer == 'glorot_orthogonal':
                inits.glorot_orthogonal(self.weight, scale=2.0)
            elif self.weight_initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            elif self.weight_initializer == 'zeros':
                inits.zeros(self.weight)
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported")

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported")

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLayerLinear(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False,
    ):
        super(TwoLayerLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish, in_embed_size=560):
        super(EmbeddingBlock, self).__init__()
        self.act = act
#         self.emb = Embedding(95, hidden_channels)
        self.emb = nn.Linear(in_embed_size, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
#         print(x.shape)
#         print(self.emb.weight.shape)
#         print(self.emb.weight.type())
#         print(x.type())
        x = self.emb(x)
#         print(x.shape)
#         print(x.type())
        x = self.act(x)
#         print(x.shape)
#         print(x.type())
        return x


class EdgeGraphConv(GraphConv):

    def message(self, x_j, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j


class SimpleInteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            middle_channels,
            num_radial,
            num_spherical,
            num_layers,
            output_channels,
            act=swish
    ):
        super(SimpleInteractionBlock, self).__init__()
        self.act = act

        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)

        self.norm = GraphNorm(hidden_channels)

        # Transformations of Bessel and spherical basis representations.
        self.lin_feature1 = TwoLayerLinear(num_radial * num_spherical ** 2, middle_channels, hidden_channels)
        self.lin_feature2 = TwoLayerLinear(num_radial * num_spherical, middle_channels, hidden_channels)

        # Dense transformations of input messages.
        self.lin = Linear(hidden_channels, hidden_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.norm.reset_parameters()

        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.lin_cat.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()

        self.final.reset_parameters()

    def forward(self, x, feature1, feature2, edge_index, batch):
        x = self.act(self.lin(x))

        feature1 = self.lin_feature1(feature1)

        h1 = self.conv1(x, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)

        feature2 = self.lin_feature2(feature2)

        h2 = self.conv2(x, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)

        h = self.lin_cat(torch.cat([h1, h2], 1))

        h = h + x
        for lin in self.lins:
            h = self.act(lin(h)) + h
        h = self.norm(h, batch)
        h = self.final(h)
        return h


class ComENet(nn.Module):
    r"""
         The ComENet from the `"ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs" <https://arxiv.org/abs/2206.08515>`_ paper.

        Args:
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`8.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`256`)
            middle_channels (int, optional): Middle embedding size for the two layer linear block. (default: :obj:`256`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`3`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`2`)
            num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
            pred (int, optional): the prediction of this network. 'C' - C shift, 'CH', 2dnmr based on CH edge, 'graph', graph level feature
    """
    def __init__(
            self,
            cutoff=8.0,
            num_layers=4,
            hidden_channels=256,
            middle_channels=64,
            num_radial=3,
            num_spherical=2,
            num_output_layers=3,
            in_embed_size =560,
            # num_solvent_class = 10,
            # c_solvent_emb_dim = 32,
            # h_solvent_emb_dim = 32,
            # use_solvent=True,
            agg_method = 'sum', 
            c_out_channels=1,
            h_out_channels=1,
            # c_out_hidden = [256, 512],
            # h_out_hidden = [256, 512],
            dropout=0.2

    ):
        super(ComENet, self).__init__()
        self.hidden_channels = hidden_channels
        self.c_out_channels = c_out_channels
        self.h_out_channels = h_out_channels
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.agg_method = agg_method
        self.dropout = dropout

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        act = swish
        self.act = act

        self.feature1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        self.emb = EmbeddingBlock(hidden_channels, act, in_embed_size =in_embed_size)

        # if self.use_solvent:
        #     self.lin_out_c = Projection(hidden_channels + c_solvent_emb_dim, c_out_channels, c_out_hidden) # DO NOT USE SOLVENT ON C, + solvent_emb_dim
        #     self.lin_out_h = Projection(hidden_channels + h_solvent_emb_dim, h_out_channels, h_out_hidden)
        #     self.c_solvent_embedding = torch.nn.Embedding(num_solvent_class, c_solvent_emb_dim)
        #     self.h_solvent_embedding = torch.nn.Embedding(num_solvent_class, h_solvent_emb_dim)
        # else:
        #     self.lin_out_c = Projection(hidden_channels, c_out_channels, c_out_hidden)
        #     self.lin_out_h = Projection(hidden_channels, h_out_channels, h_out_hidden)

        self.interaction_blocks = torch.nn.ModuleList(
            [
                SimpleInteractionBlock(
                    hidden_channels,
                    middle_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    hidden_channels,
                    act,
                )
                for _ in range(num_layers)
            ]
        )

        # ###### CHANGED TO ADD DROPOUT LAYERS AND ACTIVATION
        # self.lins = torch.nn.ModuleList()
        # for _ in range(num_output_layers):
        #     # Add a linear layer
        #     self.lins.append(Linear(hidden_channels + solvent_channel, hidden_channels + solvent_channel))
        #     # Add an activation function after the linear layer
        #     # self.lins.append(nn.Tanh())
        #     # Add a dropout layer after the activation function
        #     self.lins.append(nn.Dropout(self.dropout))
        
        # # for _ in range(num_output_layers):
        # #     self.lins.append(Linear(hidden_channels, hidden_channels))

        # # projection layer after graph pooling to predict 2d NMR
        # self.lin_out = Projection(hidden_channels + solvent_channel, c_out_channels, c_out_hidden, self.dropout)
        # self.lin_out_h = Projection(hidden_channels + solvent_channel, h_out_channels, h_out_hidden, self.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        # for lin in self.lins:
        #     if hasattr(lin, 'reset_parameters'):
        #         lin.reset_parameters()
        # self.lin_out.reset_parameters()

    def forward(self, data):
        batch = data.batch
#         z = npz_data.z.long()
        z = data.x
        pos = data.pos

        solvent_class = data.solvent_class

        num_nodes = z.size(0)
        # print(pos.shape)
        # print(num_nodes)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        j, i = edge_index

        vecs = pos[j] - pos[i]

        ### TODO 
        # deal with when i and j are the same atom (npz_data issue, or disconnected graph)
        # vecs = torch.where(vecs == 0, torch.full_like(vecs, 0.0001), vecs)

        dist = vecs.norm(dim=-1)

        # Embedding block.
        x = self.emb(z)

        ##### Embedding solvent class
        # solvent_class = self.sol_embed(solvent_class)
        # print(solvent_class.shape)

        # Calculate distances.
        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
        argmin0[argmin0 >= len(i)] = 0
        n0 = j[argmin0]
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add

        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]
        # --------------------------------------------------------

        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        # i[argmin] = range(0, num_nodes)
        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]

        # ----------------------------------------------------------

        # n0, n1 for i
        n0 = n0[i]
        n1 = n1[i]

        # n0, n1 for j
        n0_j = n0_j[j]
        n1_j = n1_j[j]

        # tau: (iref, i, j, jref)
        # when compute tau, do not use n0, n0_j as ref for i and j,
        # because if n0 = j, or n0_j = i, the computed tau is zero
        # so if n0 = j, we choose iref = n1
        # if n0_j = i, we choose jref = n1_j
        mask_iref = n0 == j
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]
        idx_iref = argmin0[i]
        idx_iref[mask_iref] = argmin1[i][mask_iref]

        mask_jref = n0_j == i
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][i],
            vecs[argmin1][i],
            vecs[idx_iref],
            vecs[idx_jref]
        )

        # Calculate angles.
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # Calculate torsions.
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()

        # print((dist_ji == 0).sum())

        # print(torch.mean(dist_ji))
        # print(dist_ji)

        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji

        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi


        # print('a, phi', torch.isnan(a).any())
        # print('b, phi', torch.isnan(b).any())

        # Calculate right torsions.
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        # print('a, tau', torch.isnan(a).any())
        # print('b, tau', torch.isnan(b).any())

        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)

        # print('dist', dist)
        # print('theta', theta)
        # print('phi', phi)
        # print('f1', feature1)
        # print('f2', feature2)
        # print('f1, ComENet', torch.isnan(feature1).any())
        # print('f2, ComENet', torch.isnan(feature2).any())
        # print('dist, ComENet', torch.isnan(dist).any())
        # print('theta, ComENet', torch.isnan(theta).any())
        # print('pi, ComENet', torch.isnan(phi).any())
        # print('tau, ComENet', torch.isnan(tau).any())



        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature1, feature2, edge_index, batch)

        # # Since graph_embeds is per graph, and node_reps is per node, we need to use 'batch' to map graph embeddings to nodes
        # solvent_class_per_node = solvent_class[batch]
        # x = torch.concat([solvent_class_per_node, x], dim=1)
        # # print('x', x.shape)

        # for lin in self.lins:
        #     x = self.act(lin(x))

        return x
    
#     def predict_c(self, x, npz_data):
#         z = npz_data.x
#         batch = npz_data.batch
            
#         # Identify carbon and hydrogen nodes
#         carbon_nodes = (z[:, 0] == 5).nonzero(as_tuple=True)[0]  # Assuming the first feature indicates the atom type
#         hydrogen_nodes = (z[:, 0] == 0).nonzero(as_tuple=True)[0]

#         # gather all c nodes and make c shift prediction
#         c_features = x[carbon_nodes]
#         out_c = self.lin_out(c_features)

#         return [out_c, None]
    
    
#     def predict_ch(self, x, npz_data):
#         z = npz_data.x
#         batch = npz_data.batch

#         # Identify carbon and hydrogen nodes
#         carbon_nodes = (z[:, 0] == 5).nonzero(as_tuple=True)[0]  # Assuming the first feature indicates the atom type
#         hydrogen_nodes = (z[:, 0] == 0).nonzero(as_tuple=True)[0]

#         c_features = x[carbon_nodes]
#         out_c = self.lin_out(c_features)

#         # for each c, gather its h features and predict h shifts
#         h_features_average = []
#         # Loop through each carbon node
#         for c_node in carbon_nodes:
#             # Find edges where the carbon node is a source
#             connected_edges = (npz_data.edge_index[0] == c_node).nonzero(as_tuple=True)[0]

#             # Find corresponding target nodes
#             connected_hydrogens = [npz_data.edge_index[1, e].item() for e in connected_edges if npz_data.edge_index[1, e] in hydrogen_nodes]

#             if len(connected_hydrogens) > 0:
#                 # Extract features for these hydrogen nodes
#                 h_node_features = x[connected_hydrogens]

#                 # Calculate the average of these features
#                 avg_h_features = torch.mean(h_node_features, dim=0)
#                 h_features_average.append(avg_h_features)
#             else:
#                 continue
            
#         h_features = torch.stack(h_features_average)
#         # print(h_features.shape)
#         # print(h_features.shape)
#         out_h = self.lin_out_h(h_features)
#         # print(out_h.shape)
#         out = [out_c, out_h]
#         return out


#     def forward(self, batch_data):
#         x = self.gnn_forward(batch_data)
#         if batch_data[0].has_h:
#             # print('predict CH')
#             out = self.predict_ch(x, batch_data)
#         else:
#             out = self.predict_c(x, batch_data)
#         return out
    
# class Projection(nn.Module):
#     def __init__(self, input_size=2048, output_size=128, hidden_sizes=[512, 512], dropout=0.2):
#         super(Projection, self).__init__()

#         layers = []
#         layers.append(nn.Linear(input_size, hidden_sizes[0]))
#         for i in range(len(hidden_sizes) - 1):
#             layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
#             layers.append(nn.Dropout(dropout))
#             layers.append(nn.ReLU())

#         # Output layer
#         layers.append(nn.Linear(hidden_sizes[-1], output_size))
#         layers.append(nn.ReLU())

#         # Combine all layers
#         self.model = nn.Sequential(*layers)

#     def reset_parameters(self):
#         for layer in self.model:
#             if hasattr(layer, 'reset_parameters'):
#                 layer.reset_parameters()


#     def forward(self, x):
#         out = self.model(x)
#         return out
    

# if __name__ == '__main__':
#     import pickle
#     from load_graph_cnmr_hnmr_alignment import graph_nmr_alignment_data, custom_collate_fn, CustomBatchSampler
#     from torch.utils.npz_data import DataLoader


#     graph_path = '/scratch0/haox/2DNMR_prediction_gt/Datasets/graph3d/'
#     nmr_path = '/scratch0/haox/yunruili/'
#     # cnmr_path = '/scratch0/haox/yunruili/cnmr_alignment'
#     # hnmr_path = '/scratch0/haox/yunruili/hnmr_alignment'
#     csv_cnmr = 'filtered_cnmr_smile_dataset_22k.csv'
#     csv_hnmr = 'filtered_hnmr_smile_dataset_67.csv'
#     csv_common = 'filtered_common_smile_dataset_1600.csv'
#     dataset_c = graph_nmr_alignment_data(csv_cnmr, graph_path, nmr_path, type='c')
#     dataset_h = graph_nmr_alignment_data(csv_hnmr, graph_path, nmr_path, type='h')
#     dataset_ch = graph_nmr_alignment_data(csv_common, graph_path, nmr_path, type='both')
#     dataloader_c = DataLoader(dataset_c, batch_size=1, shuffle= False, collate_fn=custom_collate_fn)
#     dataloader_h = DataLoader(dataset_h, batch_size=1, shuffle= False, collate_fn=custom_collate_fn)
#     dataloader_ch = DataLoader(dataset_ch, batch_size=1, shuffle= False, collate_fn=custom_collate_fn)

#     custom_loader = CustomBatchSampler(dataloader_c, dataloader_h, dataloader_ch, n1=20, n2=5)


#     model = ComENet(in_embed_size=3, out_channels=1, agg_method='sum', hidden_channels=256, c_out_hidden=[128], h_out_hidden=[128],\
#          num_layers=3, num_output_layers=2)
#     model.cuda()
    
#     for graph, cnmr, hnmr, filename in custom_loader:
#         graph = graph.cuda()
#         try:
#             c_shifts, h_shifts = model(graph)
#             if graph[0].has_h:
#                 hnmr = hnmr.cuda()
#                 if graph[0].has_c:
#                     cnmr = cnmr.cuda()
#                     loss = nn.MSELoss()(c_shifts, cnmr) + nn.MSELoss(h_shifts, hnmr)
#                 else:
#                     loss = nn.MSELoss()(h_shifts, hnmr)
#             else:
#                 loss = nn.MSELoss()(c_shifts, cnmr)
#         except Exception as e:
#             print(filename)
#             print(e)


    
    # invalid_file = []
    # for i, (graph, cnmr, hnmr, filename) in enumerate(dataloader_c):
    #     # print(filename)
    #     # print(cnmr.shape)
    #     # print(cnmr)
    #     # print(hnmr.shape)

    #     try:
    #         c_shifts, h_shifts = model(graph)
    #     except Exception as e:
    #         invalid_file.append(filename[0])

    #     # print(c_shifts.shape)
    #     # print(c_shifts)
    #     # print(h_shifts.shape)

    #     if torch.isnan(c_shifts).any(): # or torch.isnan(h_shifts).any():
    #         invalid_file.append(filename[0])
    #         continue

    #     # if hnmr.shape[0] != h_shifts.shape[0]:
    #     #     invalid_file.append(filename[0])
    #     #     continue
    #     if cnmr.shape[0] != c_shifts.shape[0]:
    #         invalid_file.append(filename[0])
    #         continue
    # print(invalid_file)
    # print(len(invalid_file))