
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch as t
from GAT_layer_v2 import GATv2Conv
from utils import train_features_choose, test_features_choose, build_heterograph
from scipy.sparse import coo_matrix
from scipy.special import factorial, comb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, embedding_size, drop_rate):
        super(MLP, self).__init__()
        self.embedding_size = embedding_size
        self.drop_rate = drop_rate

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 2, self.embedding_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 4, self.embedding_size // 6),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 6, 1, bias=False),
            nn.Sigmoid()
        ).to(device)
        self.mlp_prediction.apply(init_weights)

    def forward(self, rd_features_embedding):
        predict_result = self.mlp_prediction(rd_features_embedding)
        return predict_result

class MLP2(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        # print(input_dim, feature_dim, hidden_dim, output_dim)
        super(MLP2, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre =   nn.Linear(input_dim, feature_dim,bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out =    nn.Linear(feature_dim, output_dim,bias=True)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu=nn.PReLU().cuda()
        x = prelu(x)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

def build_sparse_graph(data_cf):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    cf = data_cf.copy()
    cf = cf.T

    cf[:, 1] = cf[:, 1] + 585  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(585+88, 585+88))
    return _bi_norm_lap(mat)

class metaCDA(nn.Module):
    def __init__(self, in_circfeat_size, in_disfeat_size, outfeature_size, heads, drop_rate, negative_slope,
                 features_embedding_size, negative_times, uiMat, edge_dropout_rate=0.1, mess_dropout_rate=0.1 ):
        super(metaCDA, self).__init__()
        self.in_circfeat_size = in_circfeat_size
        self.in_disfeat_size = in_disfeat_size
        self.outfeature_size = outfeature_size
        self.heads = heads
        self.drop_rate = drop_rate
        self.negative_slope = negative_slope
        self.features_embedding_size = features_embedding_size
        self.negative_times = negative_times
        self.LayerNums = 3
        self.uiMat = uiMat
        self.n_hops = 7
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        initializer = nn.init.xavier_uniform_
        self.user_t = initializer(torch.empty(585, 1))
        self.item_t = initializer(torch.empty(88, 1))
        self.dropout = nn.Dropout(p=mess_dropout_rate)
        self.dev = device

        uimat = self.uiMat[: 585, 585:]
        sparse_matrix = coo_matrix(uimat)
        values = torch.FloatTensor(sparse_matrix.tocoo().data)
        global indices
        indices = np.vstack((sparse_matrix.tocoo().row, sparse_matrix.tocoo().col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = sparse_matrix.tocoo().shape
        uimat1 = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.uiadj = uimat1
        self.iuadj = uimat1.transpose(0, 1)
        self.meta_netu = nn.Linear(128 * 3, 128, bias=True)
        self.meta_neti = nn.Linear(128 * 3, 128, bias=True)


        self.encoder = nn.ModuleList()
        for i in range(0, self.LayerNums):
            self.encoder.append(GCN_layer())
        self.k = 3
        k = self.k
        self.mlp = MLP2(128, 128* k, 128 // 2, 128 * k)
        self.mlp1 = MLP2(128, 128 * k, 128 // 2, 128 * k)
        self.mlp2 = MLP2(128, 128 * k, 128 // 2, 128 * k)
        self.mlp3 = MLP2(128, 128 * k, 128 // 2, 128 * k)
        self.meta_netu = nn.Linear(128 * 3, 128, bias=True)
        self.meta_neti = nn.Linear(128 * 3, 128, bias=True)




        # 图注意层（多头）
        self.att_layer = GATv2Conv(self.outfeature_size, self.outfeature_size, self.heads, self.drop_rate,
                                   self.drop_rate, self.negative_slope)


        # 定义投影算子
        self.W_rna = nn.Parameter(torch.zeros(size=(self.in_circfeat_size, self.outfeature_size)))
        self.W_dis = nn.Parameter(torch.zeros(size=(self.in_disfeat_size, self.outfeature_size)))

        # 初始化投影算子
        nn.init.xavier_uniform_(self.W_rna.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_dis.data, gain=1.414)

        # 定义卷积层的权重初始化函数
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        # 二维卷积层搭建
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 1), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 4), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer16 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 16), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer32 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 32), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)
        # 初始化
        self.cnn_layer1.apply(init_weights)
        self.cnn_layer4.apply(init_weights)
        self.cnn_layer16.apply(init_weights)
        self.cnn_layer32.apply(init_weights)

        # MLP
        self.mlp_prediction = MLP(self.features_embedding_size, self.drop_rate)

    def metaregular(self,em0,em,adj):
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[:,torch.randperm(embedding.shape[1])]
            corrupted_embedding = corrupted_embedding[torch.randperm(embedding.shape[0])]
            return corrupted_embedding
        def score(x1,x2):
            x1=F.normalize(x1,p=2,dim=-1)
            x2=F.normalize(x2,p=2,dim=-1)
            return torch.sum(torch.multiply(x1,x2),1)
        user_embeddings = em
        Adj_Norm =t.from_numpy(np.sum(adj,axis=1)).float().cuda()
        adj=self.sparse_mx_to_torch_sparse_tensor(adj)
        edge_embeddings = torch.spmm(adj.cuda(),user_embeddings)/Adj_Norm
        user_embeddings=em0
        graph = torch.mean(edge_embeddings,0)
        pos   = score(user_embeddings,graph)
        neg1  = score(row_column_shuffle(user_embeddings),graph)
        global_loss = torch.mean(-torch.log(torch.sigmoid(pos-neg1)))
        return global_loss

    def metafortansform(self, auxiembedu, targetembedu, auxiembedi, targetembedi):
        # Neighbor information of the target node
        uneighbor = t.matmul(self.uiadj.cuda(), self.ui_itemEmbedding)
        ineighbor = t.matmul(self.iuadj.cuda(), self.ui_userEmbedding)

        # Meta-knowlege extraction
        tembedu = (self.meta_netu(t.cat((auxiembedu, targetembedu, uneighbor), dim=1).detach()))
        tembedi = (self.meta_neti(t.cat((auxiembedi, targetembedi, ineighbor), dim=1).detach()))

        """ Personalized transformation parameter matrix """
        # Low rank matrix decomposition
        metau1 = self.mlp(tembedu).reshape(-1, 128, self.k)  # d*k
        metau2 = self.mlp1(tembedu).reshape(-1, self.k, 128)  # k*d
        metai1 = self.mlp2(tembedi).reshape(-1, 128, self.k)  # d*k
        metai2 = self.mlp3(tembedi).reshape(-1, self.k, 128)  # k*d
        meta_biasu = (torch.mean(metau1, dim=0))
        meta_biasu1 = (torch.mean(metau2, dim=0))
        meta_biasi = (torch.mean(metai1, dim=0))
        meta_biasi1 = (torch.mean(metai2, dim=0))
        low_weightu1 = F.softmax(metau1 + meta_biasu, dim=1)
        low_weightu2 = F.softmax(metau2 + meta_biasu1, dim=1)
        low_weighti1 = F.softmax(metai1 + meta_biasi, dim=1)
        low_weighti2 = F.softmax(metai2 + meta_biasi1, dim=1)

        # The learned matrix as the weights of the transformed network
        tembedus = (t.sum(t.multiply((auxiembedu).unsqueeze(-1), low_weightu1),
                          dim=1))  # Equal to a two-layer linear network; Ciao and Yelp data sets are plus gelu activation function
        tembedus = t.sum(t.multiply((tembedus).unsqueeze(-1), low_weightu2), dim=1)
        tembedis = (t.sum(t.multiply((auxiembedi).unsqueeze(-1), low_weighti1), dim=1))
        tembedis = t.sum(t.multiply((tembedis).unsqueeze(-1), low_weighti2), dim=1)
        transfuEmbed = tembedus
        transfiEmbed = tembedis
        return transfuEmbed, transfiEmbed

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.in_circfeat_size, 128 ))
        self.item_embed = initializer(torch.empty(self.in_disfeat_size, 128))

        # [n_users+n_items, n_users+n_items]
        adj_mat = build_sparse_graph(indices)
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(adj_mat).to(device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, graph, circ_feature_tensor, dis_feature_tensor, rel_matrix, train_model,  mess_dropout=True, edge_dropout=True):

        circ_circ_f = circ_feature_tensor.mm(self.W_rna)
        dis_dis_f = dis_feature_tensor.mm(self.W_dis)


        # for circRNA->adj
        circ_feature_tensor_cpu = circ_feature_tensor.cpu()
        matAdj_circ = np.where(circ_feature_tensor_cpu > 0.5, 1, 0)

        # for disease->adj
        dis_feature_tensor_cpu = dis_feature_tensor.cpu()
        matAdj_dis = np.where(dis_feature_tensor_cpu > 0.5, 1, 0)
        dense_matrix = rel_matrix.cpu().numpy()

        mat_a = np.concatenate((matAdj_circ, dense_matrix), axis=1)

        # 在列轴上拼接矩阵2和矩阵4
        mat_b = np.concatenate((matAdj_dis, dense_matrix.T), axis=1)

        # 在行轴上拼接矩阵A和矩阵B
        mat_c = np.concatenate((mat_a, mat_b), axis=0)
        # 转换为 CSR 稀疏矩阵
        rel_matrix_cirdcis_circdis = csr_matrix(mat_c)



        # 使用 torch.sparse_coo_tensor 函数创建稀疏矩阵

        self.ui_embeddings = torch.cat([circ_circ_f, dis_dis_f], 0)
        self.all_user_embeddings = [circ_circ_f]
        self.all_item_embeddings = [dis_dis_f]
        self.all_ui_embeddings = [self.ui_embeddings]

        circ_index = np.arange(0, 585)
        dis_index = np.arange(0, 88)
        ui_index = np.array(circ_index.tolist() + [i + 585 for i in dis_index])
        for i in range(len(self.encoder)):
            layer = self.encoder[i]
            if i == 0:
                userEmbeddings0 = layer(circ_circ_f, circ_feature_tensor, circ_index)
                itemEmbeddings0 = layer(dis_dis_f, dis_feature_tensor, dis_index)
                uiEmbeddings0 = layer(self.ui_embeddings, rel_matrix_cirdcis_circdis, ui_index)
            else:
                userEmbeddings0 = layer(userEmbeddings, circ_feature_tensor, circ_index)
                itemEmbeddings0 = layer(itemEmbeddings, dis_feature_tensor, dis_index)
                uiEmbeddings0 = layer(uiEmbeddings, rel_matrix_cirdcis_circdis, ui_index)

            # Aggregation of message features across the two related views in the middle layer then fed into the next layer
            self.ui_userEmbedding0, self.ui_itemEmbedding0 = torch.split(uiEmbeddings0, [585, 88])
            userEd = (userEmbeddings0 + self.ui_userEmbedding0) / 2.0
            itemEd = (itemEmbeddings0 + self.ui_itemEmbedding0) / 2.0
            userEmbeddings = userEd
            itemEmbeddings = itemEd
            uiEmbeddings = torch.cat([userEd, itemEd], 0)
            if 1 == 1:
                norm_embeddings = F.normalize(userEmbeddings0, p=2, dim=1)
                self.all_user_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(itemEmbeddings0, p=2, dim=1)
                self.all_item_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(uiEmbeddings0, p=2, dim=1)
                self.all_ui_embeddings += [norm_embeddings]
            else:
                self.all_user_embeddings += [userEmbeddings]
                self.all_item_embeddings += [norm_embeddings]
                self.all_ui_embeddings += [norm_embeddings]

        self.userEmbedding = t.stack(self.all_user_embeddings, dim=1)
        self.userEmbedding = t.mean(self.userEmbedding, dim=1)
        self.itemEmbedding = t.stack(self.all_item_embeddings, dim=1)
        self.itemEmbedding = t.mean(self.itemEmbedding, dim=1)
        self.uiEmbedding = t.stack(self.all_ui_embeddings, dim=1)
        self.uiEmbedding = t.mean(self.uiEmbedding, dim=1)
        self.ui_userEmbedding, self.ui_itemEmbedding = t.split(self.uiEmbedding, [585, 88])

        metatsuembed, metatsiembed = self.metafortansform(self.userEmbedding, self.ui_userEmbedding, self.itemEmbedding, self.ui_itemEmbedding)

        circ_circ_f = circ_circ_f + metatsuembed
        dis_dis_f = dis_dis_f + metatsiembed
        N = circ_circ_f.size()[0] + dis_dis_f.size()[0]  # 异构网络的节点个数,num_circ+num_dis

        # 异构网络节点的特征表达矩阵
        h_c_d_feature = torch.cat((circ_circ_f, dis_dis_f), dim=0)

        embs = [h_c_d_feature]

        agg_embed = h_c_d_feature

        for k in range(1, self.n_hops + 1):
            self._init_weight()
            interact_mat = self._sparse_dropout(self.sparse_norm_adj,
                                                self.edge_dropout_rate) if edge_dropout \
                else self.interact_mat

            side_embeddings = torch.sparse.mm(self.sparse_norm_adj, h_c_d_feature)
            user_embedds, item_embedds = torch.split(side_embeddings, [585, 88],
                                                     dim=0)
            user_embedds = user_embedds * (torch.exp(-self.user_t).to(self.dev) *
                                           torch.pow(self.user_t.to(self.dev), torch.FloatTensor([k]).to(self.dev)).to(self.dev)
                                           / torch.FloatTensor([factorial(k)]).to(self.dev))
            item_embedds = item_embedds * (torch.exp(-self.item_t).to(self.dev) *
                                           torch.pow(self.item_t.to(self.dev), torch.FloatTensor([k]).to(self.dev)).to(self.dev)
                                           / torch.FloatTensor([factorial(k)]).to(self.dev))
            side_embeddings_cur = torch.cat([user_embedds, item_embedds], dim=0)
            agg_embed = side_embeddings
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            embs.append(side_embeddings_cur)
        embs = torch.stack(embs, dim=1)
        circ_embs = embs[:585, :]
        dis_embs = embs[88:, :]

        # 特征聚合
        res = self.att_layer(graph, h_c_d_feature) + embs # size:[nodes,heads,outfeature_size]


        x = res.view(N, 1, self.heads, -1)

        cnn_embedding1 = self.cnn_layer1(x).view(N, -1)
        cnn_embedding4 = self.cnn_layer4(x).view(N, -1)
        cnn_embedding16 = self.cnn_layer16(x).view(N, -1)
        cnn_embedding32 = self.cnn_layer32(x).view(N, -1)

        cnn_outputs = torch.cat([cnn_embedding1, cnn_embedding4, cnn_embedding16, cnn_embedding32], dim=1)
        # print('features_embedding_size:', cnn_outputs.size()[1])

        if train_model:
            train_features_inputs, train_lable = train_features_choose(rel_matrix, cnn_outputs, self.negative_times)
            train_mlp_result = self.mlp_prediction(train_features_inputs)
            return train_mlp_result, train_lable
        else:
            test_features_inputs, test_lable = test_features_choose(rel_matrix, cnn_outputs)
            test_mlp_result = self.mlp_prediction(test_features_inputs)
            return test_mlp_result, test_lable

class GCN_layer(nn.Module):
    def __init__(self):
        super(GCN_layer, self).__init__()
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        if isinstance(adj, csr_matrix):
            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1))
            eps = 1e-8  # 设置一个极小的数
            d_inv_sqrt = np.power(rowsum + eps, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            result = (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()
            return result
        else:
            adj = torch.tensor(adj)
        adj_cpu = adj.cpu().numpy()
        adj = sp.coo_matrix(adj_cpu)
        rowsum = np.array(adj.sum(1))
        eps = 1e-8  # 设置一个极小的数
        d_inv_sqrt = np.power(rowsum + eps, -0.5).flatten()
        # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        result = (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()
        return result

    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = self.normalize_adj(subset_Mat)
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).cuda()
        out_features = torch.spmm(subset_sparse_tensor, subset_features)
        new_features = torch.empty(features.shape).cuda()
        new_features[index] = out_features
        dif_index = np.setdiff1d(torch.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]
        return new_features

