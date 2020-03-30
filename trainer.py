import time
import torch

from torch import nn, optim
import torch.nn.functional as F

from model import MultiStageTCN


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageTCN(num_blocks, num_layers, num_f_maps, dim, num_classes)
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

            while data_loader.has_next_test():
                number += 1
                if number % 10 == 0:
                    print(number)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("loss = %f,   acc = %f" % (epoch_loss / number, float(correct)/total))
                batch_input, batch_target, mask = data_loader.next_test_batch(batch_size)
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
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(data_loader.test_list),
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
