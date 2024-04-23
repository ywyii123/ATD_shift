import torch

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
# from .sr_model import SRModel, SR2Model
from basicsr.models.sr3_model import SR3Model

from basicsr.utils import generate_lq


@MODEL_REGISTRY.register()
class EDSR2Model(SR3Model):

    def test(self, gt_size=None):
        
        test_opt = self.opt['train']['consistency_opt']
        sigma = test_opt['sigma_max']
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