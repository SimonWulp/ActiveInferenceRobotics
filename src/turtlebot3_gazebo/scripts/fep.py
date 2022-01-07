import numpy as np
import torch
from decoder import ConvDecoder

class FEP():
    def __init__(self, environment, decoder) -> None:
        self.environment = environment
        self.decoder = decoder

        self.img_width = 256
        self.img_height = 256

        self.mu = np.empty((1, 2))
        self.s_v = np.empty((1, 1, self.img_height, self.img_width))
        self.g_mu = np.empty((1, 1, self.img_height, self.img_width))
        self.pred_error = np.empty((1,1,self.img_height, self.img_width))

        self.a = np.empty((1, 2))
        self.a_dot = np.empty((1, 2))

        self.dt = 0.02
        self.a_clp = 1

    def get_visual_forward(self, inp):
        inp = torch.tensor(inp, dtype=torch.float, requires_grad=True)
        outp = self.decoder.forward(inp)
        
        return inp, outp

    def get_dF_dmu_vis(self, inp, outp):
        neg_dF_dg = torch.tensor((1 / self.sigma) * self.pred_error)

        inp.grad = torch.zeros(input.size())
        outp.backward(neg_dF_dg, retain_graph=True)
        
        return input.grad.data.cpu().numpy()
    
    # TODO: Implement for active inference
    def get_dF_dmu_att(self):
        ...

    def get_dF_da_vis(self, dF_dmu_vis):
        return (-1) * dF_dmu_vis * self.dt

    def step(self):
        inp, outp = self.get_visual_forward(self.mu)
        self.g_mu = outp.data.cpu().numpy()
        self.pred_error = self.s_v - self.g_mu

        dF_dmu_vis = self.get_dF_dmu_vis(inp, outp)

        if self.active_inference:
            # dF/dmu with attractor:
            mu_dot = dF_dmu_vis + self.get_df_dmu_att(inp, outp)
        else:
            mu_dot = dF_dmu_vis

        self.mu = self.mu + self.dt * mu_dot

        a_dot = self.get_dF_da_visual(dF_dmu_vis)
        
        self.a = self.a + self.dt * a_dot
        self.a = np.clip(self.a, -self.a_clp, self.a_clp)