import torch

from basicsr.utils.diffusion import karras_schedule, p_sample_upsample_loop
from basicsr.utils.registry import MODEL_REGISTRY

from .sr_edm_unet_model import SREdmUNetModel

from basicsr.utils import generate_lq


@MODEL_REGISTRY.register()
class EdmUNetModel(SREdmUNetModel):

    def test(self,):
        
        consistency_opt = self.opt['train']['consistency_opt']
        sigma_max = consistency_opt['sigma_max']
        
        sigma = sigma_max
        sigma = torch.as_tensor(sigma).to(self.device)
        latent = torch.randn_like(self.lq_upsample, device=self.device)
        input = self.lq_upsample + sigma * latent.to(torch.float32)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(input, sigma=sigma, x_lr=self.lq_upsample,)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(input, sigma=sigma, x_lr=self.lq_upsample,)
            self.net_g.train()