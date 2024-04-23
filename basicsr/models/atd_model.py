import torch

from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class ATDModel(SRModel):

    def test(self):
        consistency_opt = self.opt['train']['consistency_opt']
        sigma_max = consistency_opt['sigma_max']
        sigma_max = torch.as_tensor(sigma_max).to(self.device)
        latent = torch.randn_like(self.lq, device=self.device)
        input = self.lq + sigma_max * latent.to(torch.float32)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(input, sigma_max, self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(input, sigma_max, self.lq)
            self.net_g.train()
