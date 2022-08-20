#################### Imports #########################
from __future__ import print_function, division
import argparse
import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
from torch.optim import lr_scheduler
import time
import numpy as np
from torch.autograd import Variable
from pathlib import Path
from PIL import Image
import pandas as pd
#from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from IPython.display import clear_output


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    # default to the value in environment variable `SM_MODEL_DIR`. Using args makes the script more portable.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args, _ = parser.parse_known_args()





    ################################# Training ###################################
    # ... load from args.train and args.test, train a model, write model to args.model_dir.

    ######### Unpack Args ##########
    train_dir = args.train
    model_dir = args.model_dir

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    use_cuda = args.use_cuda




    ########## GPU stuff that I don't know lol #############
    # define variables if GPU is to be used
    if torch.cuda.is_available():
        use_gpu = True
        print("Using GPU")
    else:
        use_gpu = False
    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
    Tensor = FloatTensor

    ########## Dataclass for segmentation ###########
    class CloudDataset(Dataset):
        def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True):
            super().__init__()

            # Loop through the files in red folder and combine, into a dictionary, the other bands
            self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if
                          not f.is_dir()]
            self.pytorch = pytorch

        def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):

            files = {'red': r_file,
                     'green': g_dir / r_file.name.replace('red', 'green'),
                     'blue': b_dir / r_file.name.replace('red', 'blue'),
                     'nir': nir_dir / r_file.name.replace('red', 'nir'),
                     'gt': gt_dir / r_file.name.replace('red', 'gt')}

            return files

        def __len__(self):

            return len(self.files)

        def open_as_array(self, idx, invert=False, include_nir=False):

            raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                                np.array(Image.open(self.files[idx]['green'])),
                                np.array(Image.open(self.files[idx]['blue'])),
                                ], axis=2)

            if include_nir:
                nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
                raw_rgb = np.concatenate([raw_rgb, nir], axis=2)

            if invert:
                raw_rgb = raw_rgb.transpose((2, 0, 1))

            # normalize
            return (raw_rgb / np.iinfo(raw_rgb.dtype).max)

        def open_mask(self, idx, add_dims=False):

            raw_mask = np.array(Image.open(self.files[idx]['gt']))
            raw_mask = np.where(raw_mask == 255, 1, 0)

            return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

        def __getitem__(self, idx):

            x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
            y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)

            return x, y

        def open_as_pil(self, idx):

            arr = 256 * self.open_as_array(idx)

            return Image.fromarray(arr.astype(np.uint8), 'RGB')

        def __repr__(self):
            s = 'Dataset class with {} files'.format(self.__len__())

            return s

    ########## Dataclass for segmentation ###########
    base_path = Path(train_dir)                                     #### NEW CHANGE FOR SAGE
    data = CloudDataset(base_path / 'train_red',
                        base_path / 'train_green',
                        base_path / 'train_blue',
                        base_path / 'train_nir',
                        base_path / 'train_gt')

    ####### Split the data ########
    train_ds, valid_ds = torch.utils.data.random_split(data, (6000, 2400))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)


    ################ Initialise Model ###############
    from model_dir import SegNet
    learning_rate = learning_rate
    num_classes = 2  # assuming cloud and non cloud
    num_channels = 4  # for the cloud data, for now
    model = SegNet(num_classes, n_init_features=num_channels)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # using this becuase SmokeNet did
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    ################ Model training ##############

    def train(model, train_dl, valid_dl, loss_fn, optimizer, scheduler, acc_fn, epochs=1):
        start = time.time()

        train_loss, valid_loss = [], []

        best_acc = 0.0

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train(True)  # Set trainind mode = true
                    dataloader = train_dl
                else:
                    model.train(False)  # Set model to evaluate mode
                    dataloader = valid_dl

                running_loss = 0.0
                running_acc = 0.0

                step = 0

                # iterate over data
                for x, y in dataloader:
                    x = x.cuda()
                    y = y.cuda()
                    step += 1

                    # forward pass
                    if phase == 'train':
                        # zero the gradients
                        optimizer.zero_grad()
                        outputs = model(x)
                        loss = loss_fn(outputs, y)

                        # the backward pass frees the graph memory, so there is no
                        # need for torch.no_grad in this training pass
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    else:
                        with torch.no_grad():
                            outputs = model(x)
                            loss = loss_fn(outputs, y.long())

                    # stats - whatever is the phase
                    acc = acc_fn(outputs, y)

                    running_acc += acc * dataloader.batch_size
                    running_loss += loss * dataloader.batch_size

                    if step % 100 == 0:
                        # clear_output(wait=True)
                        print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc,
                                                                                              torch.cuda.memory_allocated() / 1024 / 1024))
                        # print(torch.cuda.memory_summary())

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_acc / len(dataloader.dataset)

                clear_output(wait=True)
                print('Epoch {}/{}'.format(epoch, epochs - 1))
                print('-' * 10)
                print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
                print('-' * 10)

                train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return train_loss, valid_loss


    def acc_metric(predb, yb):
        return (predb.argmax(dim=1) == yb.cuda()).float().mean()

    ####### train
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss, valid_loss = train(model, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=epochs)


    ############################## Save model ####################################
    # ... train `model`, then save it to `model_dir`
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)









