import torch.optim as optim
import torch.utils
import torch.nn.utils

import matplotlib.pyplot as plt
import numpy as np
import argparse

from torch.utils.data import Dataset
from torchvision import datasets,transforms

# project utils
from utils.draw_model import DrawModel
from utils.config import *
from utils.utility import Variable,save_image,xrecons_grid

torch.set_default_tensor_type('torch.FloatTensor')

# modify here
PATH = 'data/full_numpy_bitmap_apple.npy'


# https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
class DatasetMNIST2(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = np.load(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)
        image = self.data[index].astype(np.uint8).reshape((28, 28, 1))
        # label = self.data[index]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image

# https://discuss.pytorch.org/t/load-300gb-across-200-npy-files-with-dataset-and-dataloader/66882
class MYDS(Dataset):
    def __init__(self, x_paths):
        self.xs = None

        print(f'loaded: {x_paths}')
        self.xs = torch.from_numpy(np.load(x_paths))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx]

# create custom dataset and put it to torch dataloader
train_dataset = DatasetMNIST2(PATH, transform=transforms.ToTensor())
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print('> dataset loaded:', PATH)

model = DrawModel(T,A,B,z_size,N,dec_size,enc_size)
optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(beta1,0.999))

print('> model initialized')

if USE_CUDA:
    print('> use CUDA')
    model.cuda()

def train():
    print('> strart training')

    avg_loss = 0
    count = 0

    for epoch in range(epoch_num):
        print('> starting epoch:', epoch)

        iterator = iter(train_data_loader)

        for data in train_data_loader: # data.shape = torch.Size([64, 1, 28, 28])
            next_it_data = iterator.next()
            next_it_data = next_it_data.type(torch.FloatTensor)

            bs = next_it_data.size()[0] # bs = 64
            tensor_var = Variable(next_it_data).view(bs, -1) # data.shape = torch.Size([64, 784])

            optimizer.zero_grad()
            loss = model.loss(tensor_var)

            avg_loss += loss.cpu().data.numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            count += 1
            if count % 100 == 0:
                print (f'epoch: {epoch} / count: {count}\t loss: {avg_loss / 100}')
                if count % 3000 == 0:
                    torch.save(model.state_dict(),'save/weights_%d.tar'%(count))
                    generate_image(count)

                avg_loss = 0

    torch.save(model.state_dict(), 'save/weights_final.tar')
    generate_image(count)


def generate_image(count):
    x = model.generate(batch_size)
    save_image(x,count)

def save_example_image():
    iterator = iter(train_data_loader)
    data = iterator.next()
    img = data.cpu().numpy().reshape(batch_size, 28, 28)
    imgs = xrecons_grid(img, B, A)
    plt.matshow(imgs, cmap=plt.cm.gray)
    plt.savefig('image/example.png')
    plt.clf()

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to dataset in .npy format")
    args = parser.parse_args()
    if args.path: PATH = args.path

if __name__ == '__main__':
    parsing()
    save_example_image()
    train()
