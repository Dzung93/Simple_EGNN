from sklearn.model_selection import train_test_split
import torch_geometric as tg

def split_tvt(data, train_size, valid_size, test_size, seed = 456):
    batch_size = 1
    d_train, d_mid = train_test_split(data, train_size=train_size)
    d_val, d_test = train_test_split(d_mid, test_size = test_size/(test_size + valid_size))
    d_load_train = tg.loader.DataLoader(d_train, batch_size=batch_size, shuffle=True)
    d_load_val = tg.loader.DataLoader(d_val, batch_size=batch_size)
    d_load_test = tg.loader.DataLoader(d_test, batch_size=batch_size)
    return d_load_train, d_load_val, d_load_test

def split_tt(data, train_size, seed = 456):
    batch_size = 1
    d_train, d_test = train_test_split(data, train_size=train_size)
    d_load_train = tg.loader.DataLoader(d_train, batch_size=batch_size, shuffle=True)
    d_load_test = tg.loader.DataLoader(d_test, batch_size=batch_size)
    return d_load_train, d_load_test