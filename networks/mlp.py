import torch
import torch.nn as nn
import torch.nn.functional as F


###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class CrossModal_CL(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(CrossModal_CL, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, anchor_feature, features, label):
        # anchor_feature:[bs,dim]
        # features:[bs,4,dim]
        # label : [bs] long
        anchor_feature = anchor_feature.unsqueeze(1)  # [bs.1.dim]
        features = features.transpose(2, 1)  # [ns,dim,4]
        # b, device = anchor_feature.shape[0], anchor_feature.device
        logits = torch.div(torch.matmul(anchor_feature, features), self.temperature)  # [bs.4]
        logits_max = logits.max(dim=-1, keepdim=True)[0]
        logits = logits - logits_max.detach()
        loss = F.cross_entropy(logits.squeeze(1), label)
        return loss

class MLP_att(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP_att, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class AttFlat(nn.Module):
    def __init__(self, hidden_size=512, flat_mlp_size=512, flat_out_size=1024, flat_glimpses=1, dropout_r=0.1):
        super(AttFlat, self).__init__()
        self.flat_glimpses = flat_glimpses

        self.mlp = MLP_att(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_out_size,
            dropout_r=dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x) #att: [bs,seq_len,1024]
        x_mask = (x_mask == 0).byte() #[bs, seq_len]
        att = att.masked_fill(
            x_mask.bool().unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1) #[bs, obj_num, 1024]

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1) #torch.sum(att[:, :, i: i + 1], dim=1)=1
            )

        x_atted = torch.cat(att_list, dim=1) #[bs, 512]
        x_atted = self.linear_merge(x_atted)

        return x_atted

class AttFlat_nofc(nn.Module):
    def __init__(self, hidden_size=512, flat_mlp_size=512, flat_out_size=1024, flat_glimpses=1, dropout_r=0.1):
        super(AttFlat_nofc, self).__init__()
        self.flat_glimpses = flat_glimpses

        self.mlp = MLP_att(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_out_size,
            dropout_r=dropout_r,
            use_relu=True
        )

        # self.linear_merge = nn.Linear(
        #     hidden_size * flat_glimpses,
        #     flat_out_size
        # )

    def forward(self, x, x_mask):
        att = self.mlp(x)  # att: [bs,seq_len,1024]
        x_mask = (x_mask == 0).byte()  # [bs, seq_len]
        att = att.masked_fill(
            x_mask.bool().unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)  # [bs, obj_num, 1024]

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)  # torch.sum(att[:, :, i: i + 1], dim=1)=1
            )

        x_atted = torch.cat(att_list, dim=1)  # [bs, 512]
        # x_atted = self.linear_merge(x_atted)

        return x_atted

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2