import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStageTCN(nn.Module):
    def __init__(self, num_stages, num_layers_pre_stage, num_features_per_layer, input_features_dim, output_feature_dim):
        super(MultiStageTCN, self).__init__()
        self.stage1 = SingleTCN(num_layers_pre_stage, num_features_per_layer, input_features_dim, output_feature_dim)
        self.stages = nn.ModuleList([SingleTCN(num_layers_pre_stage, num_features_per_layer, output_feature_dim, output_feature_dim) for s in range(num_stages - 1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleTCN(nn.Module):
    def __init__(self, num_layers_pre_stage, num_features_per_layer, input_features_dim, output_feature_dim):
        super(SingleTCN, self).__init__()
        self.num_features_per_layer = num_features_per_layer
        self.conv_1x1 = nn.Conv1d(input_features_dim, num_features_per_layer, 1)
        self.layers = nn.ModuleList([DilatedResidualLayer(2 ** i, num_features_per_layer, num_features_per_layer) for i in range(num_layers_pre_stage)])
        self.conv_out = nn.Conv1d(num_features_per_layer, output_feature_dim, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        new_mask = torch.stack([mask[:, 0, :]] * self.num_features_per_layer, dim=1)
        out = torch.stack([out, new_mask], dim=0)

        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out[0]) * mask

        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x, mask = x
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        out = (x + out) * mask[:, 0:1, :]
        out = torch.stack([out, mask], dim=0)
        return out
