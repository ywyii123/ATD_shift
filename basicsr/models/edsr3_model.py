import torch

from basicsr.utils.diffusion import karras_schedule, p_sample_loop
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
# from .sr_model import SRModel, SR2Model
from basicsr.models.sr4_model import SR4Model

from basicsr.utils import generate_lq


@MODEL_REGISTRY.register()
class EDSR3Model(SR4Model):

    def test(self, gt_size=None):
        
        consistency_opt = self.opt['train']['consistency_opt']
        multi_step = consistency_opt['multi_step']
        sigma_min = consistency_opt['sigma_min']
        sigma_max = consistency_opt['sigma_max']
        rho = consistency_opt['rho']
        # sigmas = karras_schedule(num_steps + 1, sigma_min, sigma_max, rho, device=self.device)
        # sigmas = sigmas.flip(dims=(0,))
        if multi_step == False:
            sigma = sigma_max
            sigma = torch.as_tensor(sigma).to(self.device)
            latent = torch.randn_like(self.lq_upsample, device=self.device)
            input = self.lq_upsample + sigma * latent.to(torch.float32)
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.net_g_ema(input, sigma=sigma, lq=self.lq_upsample,)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.net_g(input, sigma=sigma, lq=self.lq_upsample,)
                self.net_g.train()
        else:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = p_sample_loop(self.lq_upsample, self.lq, self.net_g_ema, num_steps=num_steps, sigmas=sigmas, )
                    # print(torch.max(self.output).item())
                    # print(torch.min(self.output).item())
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = p_sample_loop(self.lq_upsample, self.lq, self.net_g, num_steps=num_steps, sigmas=sigmas,)
                self.net_g.train()