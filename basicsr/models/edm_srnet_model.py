import torch

from basicsr.utils.diffusion import karras_schedule, p_sample_upsample_loop
from basicsr.utils.registry import MODEL_REGISTRY
from utils.img_util import generate_hq

from .sr_edm_srnet_model import SREdmSRNetModel

from basicsr.utils import generate_lq


@MODEL_REGISTRY.register()
class EdmSRNetModel(SREdmSRNetModel):

    def test(self,):
        
        consistency_opt = self.opt['train']['consistency_opt']
        sigma_max = consistency_opt['sigma_max']
        scale = self.opt['network_g']['scale']
        
        sigma = sigma_max
        sigma = torch.as_tensor(sigma).to(self.device)
        latent = torch.randn_like(self.lq, device=self.device)
        input = self.lq + sigma * latent.to(torch.float32)
        x_skip = generate_hq(input, scale).to(self.device)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(input, sigma=sigma, x_lr=self.lq, timestep=0, x_skip=x_skip)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(input, sigma=sigma, x_lr=self.lq, timestep=0, x_skip=x_skip)
            self.net_g.train()