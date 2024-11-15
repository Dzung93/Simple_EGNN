import numpy as np

from ase import io, Atom
from ase.neighborlist import neighbor_list, primitive_neighbor_list 
import torch_geometric as tg
from torch_geometric.data import Data

import torch
from tqdm import tqdm
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64

#One-hot encoding
type_encoding = {}
for Z in tqdm(range(1, 119), bar_format=bar_format):
        specie = Atom(Z)
        type_encoding[specie.symbol] = Z - 1
type_onehot = torch.eye(len(type_encoding))

#read file
def data_processing(file_name, r_max):
    entry = io.read(file_name, index = ':')

    energy = []
    for i in range (len(entry)):
        energy.append(entry[i].get_total_energy())

    dat = {'structure': [], 'e_scale':[] }
    #dat['structure', 'energy'] = {}
    for i in range(len(entry)):
        dat['structure'].append(entry[i])

    for i in range (len(entry)):
        dat['e_scale'].append(energy[i])


    #rescale energy
    y_m = np.mean(dat['e_scale'])
    y_std = np.std(dat['e_scale'])

    dat['e_scale'] = (dat['e_scale']-y_m)/y_std

    data = []
    for i in range (len(entry)):
        data.append(build_data(dat['structure'][i], dat['e_scale'][i], r_max))

    return data, torch.tensor(y_m), torch.tensor(y_std)


#extract edge vector between central and neighboring atom
def to_edge(lat, pos, r_max, pbc = (True,  True,  True)):
    temp_pos = pos.detach().cpu().numpy()
    out_device = pos.device
    out_dtype = pos.dtype
    
    temp_lat = lat.detach().cpu().numpy()

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = primitive_neighbor_list("ijS", pbc, temp_lat, temp_pos, cutoff=float(r_max), self_interaction=False, use_scaled_positions=False)

    # transform edge_src, edge_dst, edge_shift to tensor
    edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0).to(device=out_device)
    edge_shift=torch.tensor(edge_shift, dtype=out_dtype, device=out_device)

    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_vec = torch.index_select(pos, 0, edge_index[1]) - torch.index_select(pos, 0, edge_index[0])
    edge_vec = edge_vec + torch.einsum(
                    "ni,ij->nj",
                    edge_shift,
                    lat,  # remove batch dimension
                )

    edge_batch = pos.new_zeros(pos.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_len = torch.linalg.norm(edge_vec, dim=-1)
    return edge_index, edge_shift, edge_vec, edge_len

# build data
def build_data(entry, target, r_max):
    symbols = list(entry.symbols).copy()
    positions = torch.from_numpy(entry.positions.copy())
    lattice = torch.from_numpy(entry.cell.array.copy())#.unsqueeze(0)
    force = torch.from_numpy(entry.get_forces().copy())
    pbc = entry.pbc
    
    edge_index, edge_shift, edge_vec, edge_len = to_edge(lattice, positions, r_max, pbc)
    
    data = tg.data.Data(
        pos=positions, lattice=lattice, symbol=symbols,
        x=type_onehot[[type_encoding[specie] for specie in symbols]], # atom type (node attribute)
        z= type_onehot[[type_encoding[specie] for specie in symbols]],
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_vec=edge_vec, edge_len=edge_len,
        force=force,
        energy=torch.tensor(target).unsqueeze(0)
    )
    
    return data

# calculate average number of neighbors
def get_neighbors(dat):
    n = []
    for i in range(len(dat)):
        N = dat[i].pos.shape[0]
        for k in range(N):
            n.append(len((dat[i].edge_index[0] == k).nonzero()))
    return np.array(n)


