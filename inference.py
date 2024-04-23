import torch
import os
import os.path as osp
import sys
import argparse

from PIL import Image
from torchvision import transforms
from basicsr.archs.atd_pro_arch import ATD_pro
from basicsr.archs.edsr_arch import EDSRList
from basicsr.utils.diffusion import karras_schedule, p_sample_loop
from basicsr.utils.options import yaml_load


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input image or directory path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument(
            "--task",
            type=str,
            default="classical",
            choices=['classical', 'lightweight'],
            help="Task for the model. classical: for classical SR models. lightweight: for lightweight models."
            )
    parser.add_argument(
            "--model_path",
            type=str,
            default="",
            help="Path to the model file."
            )
    parser.add_argument("--sigma_max", type=float, default=1.0)
    parser.add_argument("--sigma_min", type=float, default=0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--gt_size", type=int, default=256)
    parser.add_argument("-opt", type=str)

    args = parser.parse_args()

    return args


def process_image(image_input_path, image_output_path, model, sigmas, uplist, downlist, gt_size, device):
    with torch.no_grad():
        image_input = Image.open(image_input_path).convert('RGB')
        image_input = transforms.ToTensor()(image_input).unsqueeze(0).to(device)
        image_input = image_input * 2.0 - 1.0
        num_steps = len(sigmas) - 1
        lq_ori = image_input
        noise = torch.randn_like(image_input).to(device)
        sigma_max = sigmas[0]
        image_input = image_input + noise * sigma_max
        # print(image_input.device)
        # print(lq_ori.device)
        image_output = p_sample_loop(image_input, lq_ori, model, num_steps, sigmas, uplist, downlist, gt_size)
        image_output = (image_output + 1) / 2
        image_output = image_output.clamp(0.0, 1.0)[0].cpu()
        # image_output = model(image_input).clamp(0.0, 1.0)[0].cpu()
        image_output = transforms.ToPILImage()(image_output)
        image_output.save(image_output_path)

def main():
    args = get_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = args.model_path
    sigma_max = args.sigma_max
    sigma_min = args.sigma_min
    rho = args.rho
    num_steps = args.num_steps
    opt = yaml_load(args.opt)
    opt_net = opt['network_g']
    opt_net.pop('type')
    sigmas = karras_schedule(num_steps + 1, sigma_min, sigma_max, rho, device)
    sigmas = sigmas.flip(dims=(0,))

    up_list = [4, 3, 2, 3]
    down_list = [1, 1, 1, 2]

    # model = ATD_pro(
    #             in_chans=3,
    #             embed_dim=96,
    #             depths=[4,4,4],
    #             upsampler='pixelshuffle',
    #             interpolation=None,
    #             channel_mult_emb=4,
    #             cond_lq=True,
    #             dropout=0.1,
    #             time_emb=True,
    #             block_type='naf', 
    #             up_list=up_list,
    #             down_list=down_list,
    #             res=True
    #            )  
    
    model = EDSRList(**opt_net)


    print(model)
    state_dict = torch.load(model_path, map_location=device)['params_ema']
    # print(state_dict.keys())
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if os.path.isdir(args.in_path):
        for file in os.listdir(args.in_path):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                image_input_path = osp.join(args.in_path, file)
                file_name = osp.splitext(file)
                image_output_path = os.path.join(args.out_path, file_name[0] + '_ATD_' + args.task + '_SRx' + str(args.scale) + file_name[1])
                process_image(image_input_path, image_output_path, model, sigmas, up_list, down_list, args.gt_size, device)
    else:
        if args.in_path.endswith('.png') or args.in_path.endswith('.jpg') or args.in_path.endswith('.jpeg'):
            image_input_path = args.in_path
            file_name = osp.splitext(osp.basename(args.in_path))
            image_output_path = os.path.join(args.out_path, file_name[0] + '_ATD_' + args.task + '_SRx' + str(args.scale) + file_name[1])
            process_image(image_input_path, image_output_path, model, device)


if __name__ == "__main__":
    main()