import numpy as np
import torch
from collections import Counter

class FEP():
    def __init__(self, ls_decoders, classifier, active_inference=False):
        self.ls_decoders = ls_decoders
        self.classifier = classifier
        self.active_inference = active_inference

        self.img_width = 80
        self.img_height = 80

        self.mu = np.zeros((1, 2))
        self.s_v = np.zeros((1, 1, self.img_height, self.img_width))
        self.g_mu = np.zeros((1, 1, self.img_height, self.img_width))
        self.pred_error = np.zeros((1, 1, self.img_height, self.img_width))

        self.attractor_image = np.zeros((1, 1, self.img_height, self.img_width))
        self.a = np.zeros((1, 2))
        self.a_dot = np.zeros((1, 2))

        self.sigma_mu_v = 5*1e3
        self.sigma_mu = 1*1e3
        self.beta = 1
        self.dt = 0.02
        self.a_clp = 1
        self.env_class_window = 10

        self.mu_hist = []
        self.a_hist = []
        self.env_class_hist = []

    def get_visual_forward(self, inp):
        inp = torch.tensor(inp, dtype=torch.float, requires_grad=True)
        outp = self.decoder.forward(inp)
        
        return inp, outp

    def get_dF_dmu_vis(self, inp, outp):
        neg_dF_dg = torch.tensor((1 / self.sigma_mu_v) * self.pred_error)

        inp.grad = torch.zeros(inp.size())
        outp.backward(neg_dF_dg, retain_graph=True)
        
        return inp.grad.data.cpu().numpy()
    
    def get_dF_dmu_att(self, inp, outp):
        att_error = self.attractor_image - self.g_mu

        inp.grad = torch.zeros(inp.size())
        outp.backward(torch.tensor(self.beta*att_error*(1/self.sigma_mu)), retain_graph=True)

        return inp.grad.cpu().data.numpy()

    def get_dF_da_vis(self, dF_dmu_vis):
        return (-1) * dF_dmu_vis

    def get_env_class(self, inp):
        outp = self.classifier.forward(torch.tensor(inp).unsqueeze(0).float())
        env_class = np.argmax(outp.tolist())
        
        return env_class


    def step(self):
        self.env_class_hist.append(self.get_env_class(self.s_v))
        self.cur_env_class = Counter(self.env_class_hist[-self.env_class_window:]).most_common(1)[0][0]
        self.decoder = self.ls_decoders[self.cur_env_class]

        inp, outp = self.get_visual_forward(self.mu)
        self.g_mu = outp.data.cpu().numpy()
        self.pred_error = self.s_v - self.g_mu

        dF_dmu_vis = self.get_dF_dmu_vis(inp, outp)

        if self.active_inference:
            mu_dot = dF_dmu_vis + self.get_dF_dmu_att(inp, outp)
        else:
            mu_dot = dF_dmu_vis

        self.mu = self.mu + self.dt * mu_dot
        self.mu_hist.append(self.mu)

        a_dot = self.get_dF_da_vis(dF_dmu_vis)
        
        if self.active_inference:
            self.a = self.a + self.dt * a_dot
            self.a = np.clip(self.a, -self.a_clp, self.a_clp)
            self.a_hist.append(self.a)