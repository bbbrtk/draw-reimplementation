from utils.draw_model import DrawModel
from utils.config import *
from utils.utility import save_image

import torch.nn.utils

torch.set_default_tensor_type('torch.FloatTensor')

# modify here
WEIGHTS = 'save/weights_6000.tar'

model = DrawModel(T,A,B,z_size,N,dec_size,enc_size)

if USE_CUDA:
    model.cuda()

state_dict = torch.load(WEIGHTS)
model.load_state_dict(state_dict)

def generate():
    x = model.generate(batch_size)
    save_image(x)

if __name__ == '__main__':
    generate()