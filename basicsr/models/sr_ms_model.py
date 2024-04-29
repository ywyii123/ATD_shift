import copy
import time
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, skip_schedule
from basicsr.utils.diffusion import karras_schedule
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import generate_lq
from basicsr.utils.img_util import generate_hq, generate_hq_pad
from .base_model import BaseModel
import torch.distributed as dist
import lpips


@MODEL_REGISTRY.register()
class SRMSModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRMSModel, self).__init__(opt)

        self.opt = opt
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        self.lpips_loss = lpips.LPIPS(net='vgg')
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

        # self.net_g = torch.compile(self.net_g) # torch2.0

    def init_training_settings(self):
        # self.net_g.train()
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()


        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('consistency_opt'):
            consistency_opt = train_opt['consistency_opt']
            loss_opt = {}
            loss_opt['type'] = consistency_opt['type']
            self.cri_consistency = build_loss(loss_opt).to(self.device)
        else:
            self.cri_consistency = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_consistency is None:
            raise ValueError('Both pixel, perceptual losses, consistency losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        # self.lq = data['lq'].to(self.device)
        # if 'gt' in data:
        self.gt = data['gt'].to(self.device)
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)

    def optimize_parameters(self, current_iter, tb_logger=None, log_iter=None):
        gt_size = self.opt['network_g']['gt_size']
        self.optimizer_g.zero_grad()
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            self.output = self.net_g(self.lq)
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            self.output = self.net_g(self.lq)
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # consistency loss
        if self.cri_consistency:
            consistency_opt = self.opt['train']['consistency_opt']
            up_list = consistency_opt['up_list']
            down_list = consistency_opt['down_list']
            num_steps = len(up_list)
            down_scale_ori = up_list[0] // down_list[0]
            sigma_min = consistency_opt.get('sigma_min', 0)
            sigma_max = consistency_opt.get('sigma_max', 1)
            rho = consistency_opt.get('rho', 1)
            total_iter = self.opt['train']['total_iter']
            skip = consistency_opt.get('skip', True)
            multi_step = consistency_opt.get('multi_step', False)
            lq_ori = generate_lq(self.gt, down_scale_ori).to(self.device)
            
            timestep = torch.randint(0, num_steps, size=(1,))
            cur_upscale = up_list[timestep]
            cur_downscale = down_list[timestep]
            cur_scale = cur_upscale / cur_downscale
            sigmas = karras_schedule(num_steps + 1, sigma_min, sigma_max, rho, device=self.device)
            sigmas = sigmas.flip(dims=(0,))
            index = timestep.repeat(self.gt.shape[0])
            cur_sigma = sigmas[index].reshape(self.gt.shape[0], 1, 1, 1)
            cur_lq = generate_lq(self.gt, cur_scale).to(self.device)
            lq_cond_cur = generate_hq(lq_ori, down_scale_ori).to(self.device)
            # cond_scale = down_scale_ori / cur_scale
            # if cond_scale == 1:
            #     lq_cond_cur = lq_ori
            # else:                    
            #     lq_cond_cur = generate_hq(lq_ori, cond_scale).to(self.device)
            
            input = generate_hq(cur_lq, cur_scale).to(self.device)
            input = input + torch.randn_like(input) * cur_sigma
            x_cur = input

            distiller = self.net_g(input, timestep, cur_sigma, lq=lq_cond_cur, gt_size=gt_size)
            if multi_step == False:
                if skip:
                    skip_steps = skip_schedule(current_iter, total_iter, num_steps)
                else:
                    skip_steps = 1
                if timestep + skip_steps > num_steps - 1:
                    x_next = self.gt
                    distiller_target = self.gt
                else:
                    next_sigma = sigmas[index + skip_steps].reshape(self.gt.shape[0], 1, 1, 1)
                    next_upscale = up_list[timestep + skip_steps]
                    next_downscale = down_list[timestep + skip_steps]
                    next_scale = next_upscale / next_downscale
                    lq_next = generate_lq(self.gt, next_scale).to(self.device)
                    x_next = lq_next + torch.randn_like(lq_next) * next_sigma
                    with torch.no_grad():
                        distiller_target = self.net_g(x_next, timestep + skip_steps, next_sigma, lq=lq_next, gt_size=gt_size).detach()
            else:
                x_next = self.gt
                distiller_target = self.gt

            l_consistency = self.cri_consistency(distiller, distiller_target)

            if dist.get_rank() == 0 and current_iter % log_iter == 0 and tb_logger is not None:
                # lq_tensor = ((self.lq + 1.0) / 2).clamp(0, 1)
                gt_tensor = ((self.gt + 1.0) / 2).clamp(0, 1)
                x_cur_tensor = ((x_cur + 1.0) / 2).clamp(0, 1)
                x_next_tensor = ((x_next + 1.0) / 2).clamp(0, 1)
                distiller_tensor = ((distiller + 1.0) / 2).clamp(0, 1)
                distiller_target_tensor = ((distiller_target + 1.0) / 2).clamp(0, 1)
                tb_logger.add_images('gt', gt_tensor, global_step=current_iter, dataformats='NCHW')
                tb_logger.add_images('x_cur', x_cur_tensor, global_step=current_iter, dataformats='NCHW')
                tb_logger.add_images('x_next', x_next_tensor, global_step=current_iter, dataformats='NCHW')
                tb_logger.add_images('distiller', distiller_tensor, global_step=current_iter, dataformats='NCHW')
                tb_logger.add_images('distiller_target', distiller_target_tensor, global_step=current_iter, dataformats='NCHW')
                tb_logger.add_scalar('loss', l_consistency, global_step=current_iter)

            l_total += l_consistency
            loss_dict['l_consistency'] = l_consistency

        l_total.backward()
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            self.feed_data(val_data)
            gt_size = val_data['gt'].shape[-1]
            self.test(gt_size=gt_size)

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name != 'lpips':
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
                    else:
                        opt_['loss'] = self.lpips_loss
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        self.lq = self.totensor(self.lq)
        out_dict['lq'] = self.lq.detach().cpu()
        self.output = self.totensor(self.output)
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            self.gt = self.totensor(self.gt)
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
    
    def totensor(self, input):
        input = (input + 1) / 2
        input = input.clamp(0, 1)
        return input