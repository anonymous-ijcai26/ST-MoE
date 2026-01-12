import os
from utils.brainnetwork_reader import MyNetworkReader
from torch_geometric.data import DataLoader
from model.ph_layer import MyPHlayer
# import deepdish as dd
import h5py
import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PH layer')
parser.add_argument('--nROI', type=int, default=116, help='Number of ROIs in AAL template')
parser.add_argument('--resolution', type=int, default=1000, help='Resolution in persistence landscape')
parser.add_argument('--land', type=int, default=8, help='layers in PL')
parser.add_argument('--dim_obj', type=int, default=0, help='methods in PH layer')
parser.add_argument('--lam_group', type=float, default=0.05, help='regularization parameters')
parser.add_argument('--batch', type=int, default=100, help='Batch_size')
args = parser.parse_args()

nROI = args.nROI
resolution = args.resolution
land = args.land
dim_obj = args.dim_obj
lam_group = args.lam_group
batch = args.batch
BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet"))
data_dir = os.path.join(os.getcwd(), "data")
if not os.path.exists(os.path.join(data_dir, 'TopoFeat')):
    os.makedirs(os.path.join(data_dir, 'TopoFeat'))


def save_data(subject, pl_feat, cur_feat, data_dir):
    with h5py.File(os.path.join(data_dir, 'TopoFeat', subject + '_ph' + '.h5'), 'w') as f:
        f.create_dataset('id', data=subject)
        f.create_dataset('Persistence_landscapes', data=pl_feat)
        f.create_dataset('Derivative_curves', data=cur_feat)


def main():
    # 这个MyNetworkReader里面是只有功能连接矩阵的
    dataset = MyNetworkReader(BrainNetwork_dir)
    dataset.data.y = dataset.data.y.squeeze()
    Sub_list = dataset.subject
    PH_loader = DataLoader(dataset, batch_size=batch, pin_memory=False, shuffle=False)
    PHlayer = MyPHlayer(resolution, land, dim_obj, nROI)
    seq_idx = 0
    num_iter = 0
    for data in PH_loader:
        data = data.to(device)
        # Topological layer 使用 MyPHlayer 类的实例 PHlayer 提取数据的拓扑特征。PHlayer 是一个持久性同调层，用于计算以下特征： PL_feat:
        # 持久性景观特征（Persistence Landscapes）。betti_feat: Betti数曲线特征（Betti Number Curves）。Cur_feat: 导数曲线特征（Derivative Curves）。
        PL_feat, betti_feat, Cur_feat = PHlayer(data.x, data.edge_index, data.edge_attr)
        Subject_ID = Sub_list[seq_idx:data.num_graphs + seq_idx]
        seq_idx = seq_idx + data.num_graphs
        for i, subject in enumerate(Subject_ID):
            save_data(subject, PL_feat[i].cpu().numpy(), Cur_feat[i].cpu().numpy(), data_dir)
        num_iter = num_iter + 1
        print("Batch %s completed for saving persistent features" % num_iter)


if __name__ == '__main__':
    main()
