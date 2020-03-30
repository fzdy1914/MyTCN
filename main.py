#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 5242
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--split', default='1')

args = parser.parse_args()

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 400
bz = 1
lr = 0.0005
num_epochs = 20

sample_rate = 1

vid_list_file = "./splits/train.split1.bundle"
vid_list_file_tst = "./splits/test.split1.bundle"
features_path = "./data/"
gt_path = "./groundTruth/"

mapping_file = "./splits/mapping_bf.txt"

model_dir = "./models/"

segment_file = "./test_segment.txt"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

file_ptr = open(segment_file, 'r')
lines = file_ptr.read().split('\n')[:-1]
file_ptr.close()
segments = []
for line in lines:
    segment = line.split(' ')
    segments.append(segment)

num_classes = len(actions_dict)

trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir, features_path, vid_list_file_tst, num_epochs, device, segments)
