import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
devicess = [0]

import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import warnings
from tqdm import tqdm
from torchvision import utils
from hparams import hparams as hp
from torch.autograd import Variable
from tqdm import tqdm
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from data_function import RefineData

weight_path_pytorch = 'pretrained/network-snapshot-020180.pt'
if hp.kind == 'mask':
    mask_path = 'templates/mask'
elif hp.kind == 'glasses':
    
    if hp.is_normal == True:
        mask_path = 'templates/frame_glasses'
    else:
        mask_path = 'templates/glasses'

def parse_testing_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default='checkpoint_latest.pt', help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=500000, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=10, help='Number of epochs per checkpoint')
    training.add_argument('--sample', type=int, default=4, help='number of samples during training')  

    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )

    parser.add_argument("--init-lr", type=float, default=0.002, help="learning rate")

    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')
    return parser



def test():

    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser = parse_testing_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    os.makedirs(args.output_dir, exist_ok=True)


    pwd = os.getcwd()
    from model import Semantic_Fusion_Network
    model = Semantic_Fusion_Network()

    model = torch.nn.DataParallel(model, device_ids=devicess)

    print("load model:", args.ckpt)
    print(os.path.join(pwd, args.output_dir, args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(pwd, args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])


    # torch.save({"model": model.state_dict()},os.path.join("checkpoint_xxx.pt"))


    model.cuda()




    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])




    from pytorch_stylegan2.stylegan2_infer_pytorch import infer_face
    class_generate = infer_face(weight_path_pytorch)


    test_dataset = RefineData(class_generate, mask_path, transform)
    test_loader = DataLoader(test_dataset, 
                            batch_size=1, 
                            shuffle=False,
                            pin_memory=False,
                            )

    model.eval()




    for i, batch in tqdm(enumerate(test_loader)):

        if batch == ['error']:
            continue


        latent, face_image, mask_image, gt_dirs, canonical = batch
        canonical = canonical.cuda()

        gt_dir_repeat = gt_dirs.float()
        gt_dir_repeat = gt_dir_repeat.unsqueeze(1).to(device)
        latent = latent.unsqueeze(1)
        latent = latent.repeat(1,14,1)

        origins = class_generate.generate_from_synthesis(latent,None)
        interfacegan_origins = class_generate.generate_from_synthesis(latent,gt_dir_repeat)
        outputs = model(face_image, mask_image, canonical)
        predict_images = class_generate.generate_from_synthesis(latent,outputs+gt_dir_repeat)


        os.makedirs(os.path.join(pwd,args.output_dir,'predict_images'), exist_ok=True)
        os.makedirs(os.path.join(pwd,args.output_dir,'origins'), exist_ok=True)
        os.makedirs(os.path.join(pwd,args.output_dir,'interfacegan_origins'), exist_ok=True)

        with torch.no_grad():
            utils.save_image(
                predict_images,
                os.path.join(pwd,args.output_dir,'predict_images',f"{i:04d}.png"),
                nrow=hp.row,
                normalize=hp.norm,
                range=hp.rangee,
            )
            utils.save_image(
                origins,
                os.path.join(pwd,args.output_dir,'origins',f"{i:04d}.png"),
                nrow=hp.row,
                normalize=hp.norm,
                range=hp.rangee,
            )
            utils.save_image(
                interfacegan_origins,
                os.path.join(pwd,args.output_dir,'interfacegan_origins',f"{i:04d}.png"),
                nrow=hp.row,
                normalize=hp.norm,
                range=hp.rangee,
            )





if __name__ == '__main__':
    test()
