import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import time


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, data_loader, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        # self.model.load_state_dict(torch.load(save_dir + "/epoch-20.model"))

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # optimizer.load_state_dict(torch.load(save_dir + "/epoch-20.opt"))
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            number = 0

            while data_loader.has_next():
                number += 1
                if number % 10 == 0:
                    print(number)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("loss = %f,   acc = %f" % (epoch_loss / number, float(correct)/total))
                batch_input, batch_target, mask = data_loader.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            data_loader.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(data_loader.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, data_breakfast, epoch, device, segments):
        self.model.eval()
        ans = []
        number = -1
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))

            for data in data_breakfast:
                number += 1
                data = data.transpose(1, 0).float()
                data.unsqueeze_(0)
                data = data.to(device)
                predictions = self.model(data, torch.ones(data.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()

                for i in range(len(segments[number]) - 1):
                    start = int(segments[number][i])
                    end = int(segments[number][i+1])
                    segment = {}
                    for j in range(start, end):
                        predict = predicted[j].item()
                        if predict not in segment and predict != 0:
                            segment[predict] = 1
                        elif predict != 0:
                            segment[predict] += 1
                    action_num = 0
                    for predict in segment:
                        if segment[predict] > action_num:
                            action_num = segment[predict]
                            action = predict
                    ans.append(action)

        print(len(ans))
        with open("ans_5.csv", "w") as f:
            f.write("Id,Category\n")

            for i in range(len(ans)):
                f.write(str(i) + "," + str(ans[i]) + "\n")
