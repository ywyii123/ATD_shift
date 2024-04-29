import torch

from basicsr.utils.diffusion import karras_schedule, p_sample_upsample_loop
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
# from .sr_model import SRModel, SR2Model
from .sr2_model import SR2Model

from basicsr.utils import generate_lq


@MODEL_REGISTRY.register()
class EDSRModel(SR2Model):

    def test(self, gt_size=None):
        
        consistency_opt = self.opt['train']['consistency_opt']
        multi_step = consistency_opt['multi_step']
        sigma_min = consistency_opt['sigma_min']
        sigma_max = consistency_opt['sigma_max']
        rho = consistency_opt['rho']
        up_list = consistency_opt['up_list']
        down_list = consistency_opt['down_list']
        repeat_list = consistency_opt['repeat_list']
        num_steps = len(up_list)
        sigmas = karras_schedule(num_steps + 1, sigma_min, sigma_max, rho, device=self.device)
        sigmas = sigmas.flip(dims=(0,))
        
        if multi_step == False:
            sigma = consistency_opt['sigma_max']
            sigma = torch.as_tensor(sigma).to(self.device)
            latent = torch.randn_like(self.lq, device=self.device)
            input = self.lq + sigma * latent.to(torch.float32)
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.net_g_ema(input, timestep=0, sigma=sigma, lq=self.lq, gt_size=gt_size)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.net_g(input, timestep=0, sigma=sigma, lq=self.lq, gt_size=gt_size)
                self.net_g.train()
        else:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = p_sample_upsample_loop(self.lq,
                                                         self.net_g_ema, 
                                                         num_steps=num_steps, 
                                                         sigmas=sigmas, 
                                                         up_list=up_list, 
                                                         down_list=down_list, 
                                                         gt_size=gt_size,
                                                         repeat_list=repeat_list)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = p_sample_upsample_loop(self.lq,
                                                         self.net_g, 
                                                         num_steps=num_steps, 
                                                         sigmas=sigmas, 
                                                         up_list=up_list, 
                                                         down_list=down_list, 
                                                         gt_size=gt_size,
                                                         repeat_list=repeat_list)
                self.net_g.train()