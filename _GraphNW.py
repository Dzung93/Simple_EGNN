from typing import Dict, Union
import math

import torch_geometric as tg
from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import Linear

from _Convolution import Convolution
from Besel_basis import BesselBasis
from Data_processing import to_edge

#Finding whether there is the exsisted path in Tensor product of irreducible representation (irreps) 1 and 2 that results in irreps out
def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

#Gather two modules to enable iteration
class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x

class Network(torch.nn.Module):
    r"""equivariant neural network
    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features
    irreps_hidden : `e3nn.o3.Irreps`
        representation of the hidden features
    irreps_out : `e3nn.o3.Irreps`
        representation of the output features
    irreps_node_attr : `e3nn.o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes
    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials
    layers : int
        number of gates (non linearities)
    max_radius : float
        maximum radius for the convolution
    number_of_basis : int
        number of basis on which the edge length are projected
    radial_layers : int
        number of hidden layers in the radial fully connected network
    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network
    num_neighbors : float
        typical number of nodes at a distance ``max_radius``
    num_nodes : float
        typical number of nodes in a graph
    """
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_node_attr,
        layers,
        mul,
        lmax,
        max_radius,
        number_of_basis: int = 8,
        radial_layers: int = 2,
        radial_neurons: int = 64,
        num_neighbors: int = 1,
        #num_nodes=1.,
        #reduce_output=True,
    ) -> None:
        super().__init__()
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        #self.num_nodes = num_nodes
        #self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        #Activation function for scalar Non linearity block
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }

        #Activation function for gates Non linearity block
        act_gates = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        # Contain: layers * Interaction block, each Interaction block: Convolution  -> Gate
        for _ in range(layers):
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
            irreps = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))
            

        # Output block: output is scalar, so all l>0 will be throwed away in lin2
        self.lin1 = Linear(
            irreps_in= irreps,
            irreps_out= irreps,
            internal_weights=True,
            shared_weights=True,
            )
        

        self.lin2 = Linear(
            irreps_in= irreps,
            irreps_out= self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            )       

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        if 'edge_index' in data:
            edge_src = data['edge_index'][0]  # edge source
            edge_dst = data['edge_index'][1]  # edge destination
            edge_vec = data['edge_vec']
        
        else:
            edge_index = radius_graph(data['pos'], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]

        return batch, edge_src, edge_dst, edge_vec

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """evaluate the network
        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        batch, edge_src, edge_dst, edge_vec = self.preprocess(data)
        edge_attr = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        # edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        
        #Learnable edge length embedded
        rad_basis = BesselBasis(r_max = self.max_radius, num_basis=self.number_of_basis, trainable=True) 
        cutoff = PolynomialCutoff(r_max = self.max_radius, p=6)(edge_length).unsqueeze(-1)
        edge_length_embedded = rad_basis(edge_length) * cutoff

        # embedded edge_length by ase 
        # edge_length_embedded = soft_one_hot_linspace(
        #     x=edge_length,
        #     start=0.0,
        #     end=self.max_radius,
        #     number=self.number_of_basis,
        #     basis='bessel',
        #     cutoff=False
        # )
        
        # edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and 'x' in data:
            assert self.irreps_in is not None
            x = data['x']
        else:
            assert self.irreps_in is None
            x = data['pos'].new_ones((data['pos'].shape[0], 118))

        if self.input_has_node_attr and 'z' in data:
            z = data['z']
        else:
            #assert self.irreps_node_attr == o3.Irreps("0e")
            assert self.input_has_node_attr is None
            z = data['pos'].new_ones((data['pos'].shape[0], 118))

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        x = self.lin1(x)
        x = self.lin2(x)
            
        return x

class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, vectorize: bool = False, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        #self.pool = False
        #if kwargs['reduce_output'] == True:
            #kwargs['reduce_output'] = False
            #self.pool = True
            
        super().__init__(**kwargs)

        # embed the one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)
        self.vectorize = vectorize

    # def force(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]):
    #     data = data.clone()
    
    #     def wrapper(pos: torch.Tensor) -> torch.Tensor:
    #         #nonlocal data
    #         lat = data.lattice
    #         edge_index, edge_shift, edge_vec, edge_len = to_edge(lat=lat, pos=pos, r_max=self.max_radius)
    #         data.edge_vec = edge_vec
    #         data.edge_index = edge_index
    #         data.edge_shift = edge_shift
    #         data.edge_len = edge_len
    #         out = super().forward(data)
    #         return torch.sum(out, dim=0).squeeze(-1)
        
    #     pos = data.pos
    
    #     f = torch.autograd.functional.jacobian(
    #         func=wrapper,
    #         inputs=pos,
    #         create_graph=self.training,  # needed to allow gradients of this output during training
    #         vectorize=self.vectorize,
    #         strict=True
    #     )
    #     f = f.negative()
    #     return f

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.x = F.relu(self.em(data.x))
        data.z = F.relu(self.em(data.z))
            
        E_per_atom = super().forward(data)

        E_tot = torch.sum(E_per_atom, dim =  0)

        # data = data.clone()
        def wrapper(pos: torch.Tensor) -> torch.Tensor:
            #nonlocal data
            d = data.clone()
            lat = data.lattice
            edge_index, edge_shift, edge_vec, edge_len = to_edge(lat=lat, pos=pos, r_max=self.max_radius)
            d.edge_vec = edge_vec
            d.edge_index = edge_index
            d.edge_shift = edge_shift
            d.edge_len = edge_len
            out = super(PeriodicNetwork,self).forward(d)
            return torch.sum(out, dim=0).squeeze(-1)
        
        pos = data.pos
    
        f = torch.autograd.functional.jacobian(
            func=wrapper,
            inputs=pos,
            create_graph=self.training,  # needed to allow gradients of this output during training
            vectorize=self.vectorize,
            strict=True
        )
        f = f.negative()
        return E_tot, E_per_atom, f

def visualize_layers(model):
    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['gate', 'tp', 'lin2', 'gate']))
    try: layers = model.mp.layers
    except: layers = model.layers

    num_layers = len(layers)
    num_ops = max([len([k for k in list(layers[i].first._modules.keys()) if k not in ['fc', 'alpha']])
                   for i in range(num_layers-1)])

    fig, ax = plt.subplots(num_layers, num_ops, figsize=(14,3.5*num_layers))
    for i in range(num_layers - 1):
        ops = layers[i].first._modules.copy()
        ops.pop('fc', None); ops.pop('alpha', None)
        for j, (k, v) in enumerate(ops.items()):
            ax[i,j].set_title(k, fontsize=textsize)
            v.cpu().visualize(ax=ax[i,j])
            ax[i,j].text(0.7,-0.15,'--> to ' + layer_dst[k], fontsize=textsize-2, transform=ax[i,j].transAxes)

    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['output', 'tp', 'lin2', 'output']))
    ops = layers[-1]._modules.copy()
    ops.pop('fc', None); ops.pop('alpha', None)
    for j, (k, v) in enumerate(ops.items()):
        ax[-1,j].set_title(k, fontsize=textsize)
        v.cpu().visualize(ax=ax[-1,j])
        ax[-1,j].text(0.7,-0.15,'--> to ' + layer_dst[k], fontsize=textsize-2, transform=ax[-1,j].transAxes)

    fig.subplots_adjust(wspace=0.3, hspace=0.5)

def smooth_cutoff(x):
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y

# Polynomial Cutoff

def _poly_cutoff(x: torch.Tensor, factor: float, p: float = 6.0) -> torch.Tensor:
    x = x * factor

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

    return out * (x < 1.0)

class PolynomialCutoff(torch.nn.Module):
    _factor: float
    p: float

    def __init__(self, r_max: float, p: float = 6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        p : int
            Power used in envelope function
        """
        super().__init__()
        assert p >= 2.0
        self.p = float(p)
        self._factor = 1.0 / float(r_max)

    def forward(self, x):
        """
        Evaluate cutoff function.

        x: torch.Tensor, input distance
        """
        return _poly_cutoff(x, self._factor, p=self.p)

# class Force(torch.nn.Module):
#     vectorize: bool
#     def __init__(
#         self,
#         model,   #energy model
#         max_radius,
#         vectorize: bool = False,
#     ):
#         super().__init__()
#         self.model = model
#         self.max_radius = max_radius
#         self.vectorize = vectorize

#     def forward(self, data):
#         data = data.clone()
    
#         def wrapper(pos: torch.Tensor) -> torch.Tensor:
#             #nonlocal data
#             lat = data.lattice
#             edge_index, edge_shift, edge_vec, edge_len = to_edge(lat=lat, pos=pos, r_max=max_radius)
#             data.edge_vec = edge_vec
#             data.edge_index = edge_index
#             data.edge_shift = edge_shift
#             data.edge_len = edge_len
#             out, _ = model(data)  
#             return out.squeeze(-1)
        
#         pos = data.pos
    
#         f = torch.autograd.functional.jacobian(
#             func=wrapper,
#             inputs=pos,
#             create_graph=self.training,  # needed to allow gradients of this output during training
#             vectorize=self.vectorize,
#             strict=True
#         )
#         f = f.negative()
#         return f