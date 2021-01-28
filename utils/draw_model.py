import torch
import torch.nn as nn
from utils.utility import *
import torch.functional as F

class DrawModel(nn.Module):
    def __init__(self,T,A,B,z_size,N,dec_size,enc_size):
        super(DrawModel,self).__init__()
        self.T = T # sequence length
        # self.batch_size = batch_size
        self.A = A
        self.B = B
        self.z_size = z_size #qsampler
        self.N = N # width of the attention window 
        self.dec_size = dec_size
        self.enc_size = enc_size
        self.cs = [0] * T # canvases * seq len
        self.logsigmas,self.sigmas,self.mus = [0] * T,[0] * T,[0] * T # helpers

        self.encoder = nn.LSTMCell(2 * N * N + dec_size, enc_size) # LSTM layer
        # self.encoder_gru = nn.GRUCell(2 * N * N + dec_size, enc_size)
        self.mu_linear = nn.Linear(dec_size, z_size)
        self.sigma_linear = nn.Linear(dec_size, z_size)

        self.decoder = nn.LSTMCell(z_size,dec_size)
        # self.decoder_gru = nn.GRUCell(z_size,dec_size)
        self.dec_linear = nn.Linear(dec_size,5)
        self.dec_w_linear = nn.Linear(dec_size,N*N)

        self.sigmoid = nn.Sigmoid()


    def compute_mu(self,g,rng,delta):
        rng_t,delta_t = align(rng,delta)
        tmp = (rng_t - self.N / 2 - 0.5) * delta_t
        tmp_t,g_t = align(tmp,g)
        mu = tmp_t + g_t
        return mu

    # DOWNSAMPLING
    def filterbank(self,gx,gy,sigma2,delta):
        rng = Variable(torch.arange(0,self.N).view(1,-1))
        mu_x = self.compute_mu(gx,rng,delta)
        mu_y = self.compute_mu(gy,rng,delta)

        a = Variable(torch.arange(0,self.A).view(1,1,-1))
        b = Variable(torch.arange(0,self.B).view(1,1,-1))

        mu_x = mu_x.view(-1,self.N,1)
        mu_y = mu_y.view(-1,self.N,1)
        sigma2 = sigma2.view(-1,1,1)

        # FX  and FY each specify N 1-dimensional Gaussian bumps
        # they're identical except that they are translated (mean-shifted) from each other at intervals of δ pixels
        Fx = self.filterbank_matrices(a,mu_x,sigma2)
        Fy = self.filterbank_matrices(b,mu_y,sigma2)

        return Fx,Fy

    def forward(self,x):
        self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size,self.dec_size))
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))

        enc_state = Variable(torch.zeros(self.batch_size,self.enc_size))
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))
        for t in list(range(self.T)):
            # from https://blog.evjang.com/2016/06/understanding-and-implementing.html
            c_prev = Variable(torch.zeros(self.batch_size,self.A * self.B)) if t == 0 else self.cs[t-1]

            x_hat = x - self.sigmoid(c_prev)     
            r_t = self.read(x,x_hat,h_dec_prev)
            h_enc_prev,enc_state = self.encoder(torch.cat((r_t,h_dec_prev),1),(h_enc_prev,enc_state))

            z,self.mus[t],self.logsigmas[t],self.sigmas[t] = self.sampleQ(h_enc_prev)
            h_dec,dec_state = self.decoder(z, (h_dec_prev, dec_state))

            self.cs[t] = c_prev + self.write(h_dec) # DYNAMIC REFINEMTNT - improve image gradually
            h_dec_prev = h_dec # RECURRENT!


    def loss(self,x):
        self.forward(x)

        criterion = nn.BCELoss() # measures the Binary Cross Entropy between the target and the output:
        x_recons = self.sigmoid(self.cs[-1]) # last elem from sequence of canvases
        Lx = criterion(x_recons,x) * self.A * self.B # multiply by width and height
        Lz = 0
        kl_terms = [0] * T # Kullback–Leibler divergence setup
        # KL formula measures how one probability distribution is different from a second

        for t in list(range(self.T)):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]

            kl_terms[t] = 0.5 * torch.sum(mu_2+sigma_2-2 * logsigma,1) - self.T * 0.5 # Eric Jang formula
            Lz += kl_terms[t]

        Lz = torch.mean(Lz) 

        # the sum of a reconstruction loss (how good the picture looks) 
        # and a latent loss for our choice of z (a measure of how bad our variational approximation is of the true latent distribution) 
        loss = Lz + Lx 
        return loss


    # correct
    def filterbank_matrices(self,a,mu_x,sigma2,epsilon=1e-9):
        t_a,t_mu_x = align(a,mu_x)
        temp = t_a - t_mu_x
        temp,t_sigma = align(temp,sigma2)
        temp = temp / (t_sigma * 2)
        F = torch.exp(-torch.pow(temp,2))
        F = F / (F.sum(2,True).expand_as(F) + epsilon)
        return F


    def attn_window(self,h_dec):
        params = self.dec_linear(h_dec)
        gx_,gy_,log_sigma_2,log_delta,log_gamma = params.split(1,1)  #21

        # the whole grid is centered at (gY,gX)
        gx = (self.A + 1) / 2 * (gx_ + 1)    # 22
        gy = (self.B + 1) / 2 * (gy_ + 1)    # 23
        delta = (max(self.A,self.B) - 1) / (self.N - 1) * torch.exp(log_delta)  # 24
        sigma2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx,gy,sigma2,delta),gamma

    # READING WITH ATTN
    def read(self,x,x_hat,h_dec_prev):
        (Fx,Fy),gamma = self.attn_window(h_dec_prev) # get a window
        def filter_img(img,Fx,Fy,gamma,A,B,N):
            Fxt = Fx.transpose(2,1) # convert filterbanks
            img = img.view(-1,B,A)

            glimpse = Fy.bmm(img.bmm(Fxt))
            glimpse = glimpse.view(-1,N*N)
            return glimpse * gamma.view(-1,1).expand_as(glimpse)

        x = filter_img(x,Fx,Fy,gamma,self.A,self.B,self.N)  # batch x (read_n*read_n)
        x_hat = filter_img(x_hat,Fx,Fy,gamma,self.A,self.B,self.N)
        return torch.cat((x,x_hat),1)  # concat along feature axis

    # 
    def write(self,h_dec=0):
        w = self.dec_w_linear(h_dec)
        w = w.view(self.batch_size,self.N,self.N)

        (Fx,Fy),gamma = self.attn_window(h_dec) # get e window
        Fyt = Fy.transpose(2,1)
        # wr = matmul(Fyt,matmul(w,Fx))
        wr = Fyt.bmm(w.bmm(Fx))
        wr = wr.view(self.batch_size,self.A*self.B)
        return wr / gamma.view(-1,1).expand_as(wr)

    # sample z_t
    def sampleQ(self,h_enc):
        e = self.normalSample()

        mu = self.mu_linear(h_enc)           # 1
        log_sigma = self.sigma_linear(h_enc) # 2
        sigma = torch.exp(log_sigma)

        return mu + sigma * e , mu , log_sigma, sigma

    def normalSample(self):
        return Variable(torch.randn(self.batch_size,self.z_size))


    def generate(self,batch_size=64):
        self.batch_size = batch_size
        h_dec_prev = Variable(torch.zeros(self.batch_size,self.dec_size),volatile = True)
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size),volatile = True)

        for t in list(range(self.T)):
            c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)) if t == 0 else self.cs[t - 1]
            z = self.normalSample() # sample
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state)) # decode from sample and prev states
            self.cs[t] = c_prev + self.write(h_dec) # write on canvas
            h_dec_prev = h_dec # recurrent 

        imgs = []
        for img in self.cs:
            imgs.append(self.sigmoid(img).cpu().data.numpy()) # arr of imgs
        return imgs
