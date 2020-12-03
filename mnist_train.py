import torch.optim as optim
from torchvision import datasets,transforms
import torch.utils
from draw_model import DrawModel
from config import *
from utility import Variable,save_image,xrecons_grid
import torch.nn.utils
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

torch.set_default_tensor_type('torch.FloatTensor')

"""
1. try https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
2. https://stackoverflow.com/a/59661024
3. 
"""

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

train_dataset2 = DatasetMNIST2('train_1_28_28.npy', transform=transforms.ToTensor())

class MyDataset(Dataset):
    def __init__(self, np_file_paths):
        self.files = np_file_paths
    
    def __getitem__(self, index):
        x = np.load(self.files[index])
        x = torch.from_numpy(x).float()
        return x
    
    def __len__(self):
        return len(self.files)


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




# img_array = np.load('full_numpy_bitmap_square.npy')
# img_array = np.load('train_1_28_28.npy', 
# allow_pickle=True, 
# encoding='latin1'
# )

# tensor_x = torch.from_numpy(img_array) # transform to torch tensor
# my_dataset = torch.utils.data.TensorDataset(tensor_x) # create your datset

# dset = MYDS('train_1_28_28.npy')

train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=batch_size, shuffle=False)

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data/', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor()])),
#     batch_size=batch_size, shuffle=False)


# for d, e in train_loader:
#     print(train_loader.__class__)
#     print(d.__class__)
#     print(d.shape)
#     break

# print('---------')

# for d in train_loader2:
#     print(train_loader2.__class__)
#     print(d.__class__)
#     break

print('---------')

model = DrawModel(T,A,B,z_size,N,dec_size,enc_size)
optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(beta1,0.999))

if USE_CUDA:
    model.cuda()

def train():
    avg_loss = 0
    count = 0
    for epoch in range(epoch_num):
        train_iter = iter(train_loader2)
        for data in train_loader2: # data.shape = torch.Size([64, 1, 28, 28])
            data_it = train_iter.next()
            data_it = data_it.type(torch.FloatTensor)

            # print(data_it.__class__)
            # print(data_it.shape)

            # print('data 0')
            # print(data_it[0].__class__)
            # print(data_it[0].shape)

            bs = data_it.size()[0] # bs = 64
            # print('var')
            d2 = Variable(data_it).view(bs, -1) # data.shape = torch.Size([64, 784])
            # print(d2)
            # print(d2.__class__)
            # print(d2.shape)
            # print(d2.dtype)
            # print(d2.hc)

            optimizer.zero_grad()
            loss = model.loss(d2)
            # print('LOSS', loss)
            avg_loss += loss.cpu().data.numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            count += 1
            if count % 100 == 0:
                print ('Epoch-{}; Count-{}; loss: {};'.format(epoch, count, avg_loss / 100))
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
    train_iter = iter(train_loader2)
    data = train_iter.next()
    img = data.cpu().numpy().reshape(batch_size, 28, 28)
    imgs = xrecons_grid(img, B, A)
    plt.matshow(imgs, cmap=plt.cm.gray)
    plt.savefig('image/example.png')

if __name__ == '__main__':
    save_example_image()
    train()