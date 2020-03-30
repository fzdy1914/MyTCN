import torch
from trainer import Trainer
from my_data_loader import MyDataLoader
import os
import argparse
import random

from read_datasetBreakfast import read_mapping_dict, load_one_data, load_test_segments, load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 5242
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')

args = parser.parse_args()

num_stages = 4
num_layers_per_stage = 10
num_features_per_layer = 64
input_features_dim = 400
batch_size = 1
lr = 0.0005
num_epochs = 50

COMP_PATH = ''

''' 
training to load train set
test to load test set
'''
train_split = os.path.join(COMP_PATH, 'splits/train.split1.bundle') #Train Split
test_split = os.path.join(COMP_PATH, 'splits/test.split1.bundle') #Test Split
GT_folder = os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video
DATA_folder = os.path.join(COMP_PATH, 'data/') #Frame I3D features for all videos
mapping_loc = os.path.join(COMP_PATH, 'splits/mapping_bf.txt')

model_folder = os.path.join(COMP_PATH, './models/')
test_segment_loc = os.path.join(COMP_PATH, './test_segment.txt')

actions_dict = read_mapping_dict(mapping_loc)

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

num_classes = len(actions_dict)

trainer = Trainer(num_stages, num_layers_per_stage, num_features_per_layer, input_features_dim, num_classes)
if args.action == "train":
    data_breakfast, labels_breakfast = load_one_data(train_split, actions_dict, GT_folder, DATA_folder, datatype='training')
    data_loader = MyDataLoader(actions_dict, data_breakfast, labels_breakfast)
    trainer.train(model_folder, data_loader, num_epochs=num_epochs, batch_size=batch_size, learning_rate=lr, device=device)

if args.action == "predict":
    data_breakfast = load_data(test_split, actions_dict, GT_folder, DATA_folder, datatype='test')
    segments = load_test_segments(test_segment_loc)
    trainer.predict(model_folder, data_breakfast, num_epochs, device, segments)
