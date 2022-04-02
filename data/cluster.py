import itertools

from sklearn.cluster import DBSCAN, OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
from scipy.sparse import vstack
import numpy as np
from numpy import prod
import torch


class LatentVariableDataset(torch.utils.data.Dataset):
    '''latent variables dataset based on the infered latent variables'''
    def __init__(self, data_dict, only_z_what=False):
        
        self.model_name = data_dict['model_name']
        self.source_dataset_name = data_dict['source_dataset_name']
        # size = data_dict['labels'].shape[0]
        # trn_size = int(size * 0.8)
        # if train:
        #     data_dict = {k: v[:trn_size] for k, v in data_dict.items()}
        # else:
        #     data_dict = {k: v[trn_size:] for k, v in data_dict.items()}
        self.dataset_size = data_dict['labels'].shape[0]
        self.only_z_what = only_z_what
        
        self.z_whats = data_dict['z_what']
        self.labels = data_dict['labels']
        if self.model_name not in ['MWS', 'VAE']:
            self.z_press = data_dict['z_pres']
            self.z_wheres = data_dict['z_where']
        
        self.num_classes = len(set(self.labels.numpy()))
        print(f'{self.source_dataset_name} has {self.num_classes} classes.')

    def __getitem__(self, index):
        if self.model_name not in ['MWS', 'VAE'] and not self.only_z_what:
            wt, wr = (self.z_whats[index], self.z_wheres[index])
            x = torch.cat([wt.flatten(), wr.flatten()], dim=-1)
        else:
            wt = self.z_whats[index]
            x = wt.flatten()
        return x, self.labels[index]

    def __len__(self):
        return self.dataset_size

class ClusteredDataset(torch.utils.data.Dataset):
    '''Clustered dataset based on the latent variables'''
    def __init__(self, data_dict, train=True):
        self.size = data_dict['labels'].shape[0]
        trn_size = int(self.size * 0.8)
        if train:
            data_dict = {k: v[:trn_size] for k, v in data_dict.items()}
        else:
            data_dict = {k: v[trn_size:] for k, v in data_dict.items()}
        
        self.memoids_data = data_dict['memoids_data']
        self.labels = data_dict['labels']
        self.wt_medoids = data_dict['wt_medoids']
        self.wt_clusters = data_dict['wt_clusters']
        self.wr_medoids = data_dict['wr_medoids']
        self.wr_clusters = data_dict['wr_clusters']

    def __getitem__(self, index):
        return self.memoids_data[index], self.labels[index]
        
    def __len__(self):
        return self.size


def get_lv_data_loader(model_name, guide, dataloaders, dataset_name, 
                       remake_data=False, args=None, only_z_what=False, 
                       trned_ite=0):
    batch_size = dataloaders[0].batch_size
    latent_data_loaders = []

    for i, split in enumerate(['train', 'test']):

        trans_z_what = only_z_what
        if trans_z_what:
            lv_path = (f"./data/latent_dataset/{dataset_name}-by-{model_name}"+\
                    f"-transWhat-it{trned_ite}-{split}.pt")
        else:
            lv_path = (f"./data/latent_dataset/{dataset_name}-by-{model_name}"+\
                    f"-NoTransWhat-it{trned_ite}-{split}.pt")

        try:
            if remake_data: raise FileNotFoundError # remake dataset
            data_dict = torch.load(lv_path)
        except FileNotFoundError:
            make_lv_dataset(lv_path, guide, dataloaders[i], model_name, 
                            dataset_name, args, trans_z_what)
            data_dict = torch.load(lv_path)
    
        dataset = LatentVariableDataset(data_dict, only_z_what=only_z_what)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                shuffle=True,)
        latent_data_loaders.append(dataloader)

    return latent_data_loaders

def make_lv_dataset(lv_path, guide, dataloader, 
                    model_name='ssp', dataset_name=None, args=None, 
                    trans_z_what=False):
    from util import transform_z_what
    z_press = []
    z_whats = []
    z_wheres = []
    labels = []
    model_type = args.model_type
    if model_type not in ['MWS', 'VAE']:
        max_strks = guide.max_strks
        if trans_z_what:
            pts_per_strk = guide.pts_per_strk
    
    print("===> Generating the latent variables")
    for imgs, ys in dataloader:
        # preprocessing
        if model_type == 'MWS':
            imgs = imgs.squeeze(1)
            obs_id = ys
        bs = imgs.shape[0]
        imgs = imgs.cuda()

        # getting the LVs
        if model_type == 'MWS':
            latent_dist = guide.get_latent_dist(imgs.round())
            # [ptcs, bs, 10, 2]
            zs = guide.sample_from_latent_dist(latent_dist, 1)
            zs = zs.view(bs, -1).type(torch.float)
            z_whats.append(zs.detach().cpu())
            labels.append(ys.detach().cpu())
        else:
            zs = guide(imgs).z_smpl
            if model_type != 'VAE':
                z_pres, z_what, z_where = zs

                if trans_z_what:
                    z_what = transform_z_what(
                                z_what.view(bs, max_strks, pts_per_strk, 2),
                                z_where.view(bs, max_strks, -1),
                                z_where_type=args.z_where_type)

                z_what = (z_what.reshape(bs, max_strks, -1) * 
                        z_pres.view(bs, max_strks, -1))
                z_where = (z_where.view(bs, max_strks, -1) *
                        z_pres.view(bs, max_strks, -1))
                z_press.append(z_pres.squeeze(0).detach().cpu())
                z_whats.append(z_what.detach().cpu())
                z_wheres.append(z_where.detach().cpu())
                labels.append(ys.detach().cpu())
            else:
                z_whats.append(zs.squeeze(0))
        
    print("===> Done generating the latent variables")

    # organize and save
    z_whats = torch.concat(z_whats, dim=0)
    labels = torch.concat(labels, dim=0)
    latent_variable_dict = {
                            "model_name": model_name,
                            "source_dataset_name": dataset_name,
                            "z_what": z_whats,
                            "labels": labels
                        }
    if model_name not in ['MWS', 'VAE']:
        z_press = torch.concat(z_press, dim=0)
        z_wheres = torch.concat(z_wheres, dim=0)
        latent_variable_dict.update({
            "z_pres": z_press,
            "z_where": z_wheres,
        })

    torch.save(latent_variable_dict, lv_path)
    
def get_clustered_data_loader(model_name, guide, dataloaders, remake_data=False):
    '''
    Args:
        model_name::str: named used to find the dataset
        guide::Guide: used to generate the latent variables representations
        dataloaders::tuple: training and validation dataloaders of the source
            dataset
    Return:
        train and validation dataloader.
    '''
    lv_clustered_path = (f"./data/latent_clustered_dataset/{model_name}.pt")
    lv_path = (f"./data/latent_dataset/{model_name}.pt")
    batch_size = dataloaders[0].batch_size
    
    # load the dataset dict or make it now
    if remake_data:
        make_medoid_dataset(lv_clustered_path, lv_path, guide, dataloaders)
    try:
        data_dict = torch.load(lv_clustered_path)
    except FileNotFoundError:
        make_medoid_dataset(lv_clustered_path, lv_path, guide, dataloaders)
        data_dict = torch.load(lv_clustered_path)

    trn_dataset = ClusteredDataset(data_dict, train=True)
    val_dataset = ClusteredDataset(data_dict, train=False)

    trn_dataloader = torch.utils.data.DataLoader(trn_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle=True,)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle=False,)
    return trn_dataloader, val_dataloader

def make_medoid_dataset(dataset_path, lv_path, guide, dataloaders, re_infer=False):
    '''make a dataset and save it
    Args:
        dataset_path::str: for saving
        lv_path::str: for loading and saving
        guide::Guide: for generating the latent variable
        dataloaders::tuple: training and validation dataloader as a source
        re_infer::bool: whether to regenerate the lv datadict
    '''
    print("===> Start making the medoid dataset dict")
    if re_infer:
        make_lv_dataset(lv_path, guide, dataloaders)
    try:
        lv_dict = torch.load(lv_path)
    except FileNotFoundError:
        make_lv_dataset(lv_path, guide, dataloaders)
        lv_dict = torch.load(lv_path)

    z_press, z_whats, z_wheres, labels = lv_dict.values()
    shp, ndim = z_whats.shape[:2], z_whats.shape[-1]
    z_where_dim = z_wheres.shape[-1]
    max_strks = guide.max_strks

    # sending in [bs*n_strks, 10 (n_ptc_per_strk * 2)]
    wr_medoids_data, wr_medoids, wr_clusters = get_medoids(
                                z_wheres.view(prod(shp), z_where_dim).numpy(),
                                z_press.numpy())
    wt_medoids_data, wt_medoids, wt_clusters = get_medoids(
                                        z_whats.view(prod(shp), ndim).numpy(),
                                        z_press.numpy())

    # [dataset_size, max_strks (eg. 4) * 10 (pts * 2)]
    wr_medoids_data = torch.tensor(wr_medoids_data
                                            ).view(-1, max_strks * z_where_dim)
    wt_medoids_data = torch.tensor(wt_medoids_data
                                            ).view(-1, max_strks * ndim)
    memoids_data = torch.concat([wr_medoids_data, wt_medoids_data], dim=-1)

    print("===> Done making the dataset dict")
    
    dataset_dict = {"memoids_data": memoids_data,
                    "labels": labels,
                    "wt_medoids": wt_medoids,
                    "wt_clusters": wt_clusters,
                    "wr_medoids": wr_medoids,
                    "wr_clusters": wr_clusters}
    torch.save(dataset_dict, dataset_path)


def get_medoids(X, press):
    '''Only the vectors kept by press are used in clustering
    Args:
        X::np.array: [num_datapoints, n_dim_features] data matrix
        press::np.array: [num_datapoints,]
    Return:
        X::np.array: [num_datapoints, n_dim_features]
        medoids: [num_clusteres, n_dim_features]
        clusters: [num_datapoints] cluster assignments for data with z_pres=1.
            -1 indicates outliers, which doesn't have a cluster.
    '''
    
    # observations used in clustering
    X_used = X[press==1]
    
    print("====> Begin clustering")
    # clustering = OPTICS(n_jobs=-1).fit(X_used)
    neigh = NearestNeighbors(radius=1.5, n_jobs=-1)
    neigh.fit(X_used)

    distantce_matrix = []
    for i,x_used in enumerate(np.array_split(X_used, 10)):
        print(i)
        A = neigh.radius_neighbors_graph(x_used, mode='distance')
        # A = A.toarray()
        distantce_matrix.append(A)
    A = vstack(distantce_matrix)

    clustering = DBSCAN(n_jobs=-1).fit(A)
    breakpoint()
    clusters = clustering.labels_
    print("====> Clustering finish\n")
    
    # partition X by clusters
    A = np.hstack((X_used, clusters[:,None]))
    A = A[A[:, -1].argsort()]
    # list of datapoints in each cluster of len number of clusters + 1
    A = np.split(A[:,:-1], 
                 np.unique(A[:, -1], return_index=True)[1][1:])
    # medoid for each cluster
    medoids = [KMedoids(n_clusters=1).fit(a).cluster_centers_[0] for a in A]
    medoids = np.vstack(medoids,)

    # new dataset array from medoids
    new_X = np.take(medoids, clusters, axis=0)

    # for the outliers, assign them their original features
    new_X[clusters == -1] = X_used[clusters == -1]

    # for the datapoints used in clustering, assign them their medoids
    X[press==1] = new_X

    return X, medoids[1:], clusters