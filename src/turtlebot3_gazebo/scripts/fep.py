import numpy as np
import torch

class FEP():
    def __init__(self, decoder, classifier):
        self.decoder = decoder
        self.classifier = classifier

        self.attractor = False

        self.img_width = 80
        self.img_height = 80

        self.mu = np.empty((1, 2))
        self.s_v = np.empty((1, 1, self.img_height, self.img_width))
        self.g_mu = np.empty((1, 1, self.img_height, self.img_width))
        self.pred_error = np.empty((1, 1, self.img_height, self.img_width))

        self.a = np.empty((1, 2))
        self.a_dot = np.empty((1, 2))

        self.sigma = 1 * 1e3
        self.dt = 0.01
        self.a_clp = 1

    def get_visual_forward(self, inp):
        inp = torch.tensor(inp, dtype=torch.float, requires_grad=True)
        outp = self.decoder.forward(inp)
        
        return inp, outp

    def get_dF_dmu_vis(self, inp, outp):
        neg_dF_dg = torch.tensor((1 / self.sigma) * self.pred_error)

        inp.grad = torch.zeros(inp.size())
        outp.backward(neg_dF_dg, retain_graph=True)
        
        return inp.grad.data.cpu().numpy()
    
    # TODO: Implement for active inference
    def get_dF_dmu_att(self, inp, outp):
        att_error = self.attractor_im - self.g_mu

        inp.grad = torch.zeros(inp.size())
        outp.backward(torch.tensor(self.beta*att_error*(1/self.sigma_mu)), retain_graph=True)

        return input.grad.cpu().data.numpy()

    def get_dF_da_vis(self, dF_dmu_vis):
        return (-1) * dF_dmu_vis * self.dt

    # TODO: implement for context classification
    def get_env_class(self, inp):
        outp = self.classifier.forward(torch.tensor(inp).unsqueeze(0).float())
        env_class = np.argmax(outp.tolist())
        
        return env_class


    def step(self):
        inp, outp = self.get_visual_forward(self.mu)
        self.g_mu = outp.data.cpu().numpy()
        self.pred_error = self.s_v - self.g_mu

        dF_dmu_vis = self.get_dF_dmu_vis(inp, outp)

        if self.attractor:
            # dF/dmu with attractor:
            mu_dot = dF_dmu_vis + self.get_dF_dmu_att(inp, outp)
        else:
            mu_dot = dF_dmu_vis

        self.mu = self.mu + self.dt * mu_dot
        self.mu = np.clip(self.mu, 0, 1)

        # a_dot = self.get_dF_da_vis(dF_dmu_vis)
        
        # self.a = self.a + self.dt * a_dot
        # self.a = np.clip(self.a, -self.a_clp, self.a_clp)


    def run(self, start, goal, iterations):
        self.mu = start
        self.s_v = goal

        path = []

        for i in range(iterations):
            path.append(self.step())

        return path