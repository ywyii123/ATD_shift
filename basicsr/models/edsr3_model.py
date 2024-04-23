import torch

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
# from .sr_model import SRModel, SR2Model
from basicsr.models.sr4_model import SR4Model

from basicsr.utils import generate_lq


@MODEL_REGISTRY.register()
class EDSR3Model(SR4Model):

    def test(self, gt_size=None):
        
        test_opt = self.opt['train']['consistency_opt']
        sigma = test_opt['sigma_max']
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