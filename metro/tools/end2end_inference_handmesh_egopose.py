"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

End-to-end inference codes for
3D hand mesh reconstruction from an image


Usage:
conda activate metro-hand
    python ./metro/tools/end2end_inference_handmesh_egopose.py  \
        --resume_checkpoint ./models/metro_release/metro_hand_state_dict.bin
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import json
import torch
import torchvision.models as models
import numpy as np
import cv2
from metro.modeling.bert import BertConfig, METRO
from metro.modeling.bert import METRO_Hand_Network as METRO_Network
from metro.modeling._mano import MANO, Mesh
from metro.modeling.hrnet.hrnet_cls_net import get_cls_net
from metro.modeling.hrnet.config import config as hrnet_config
from metro.modeling.hrnet.config import update_config as hrnet_update_config
import metro.modeling.data.config as cfg

from metro.utils.logger import setup_logger
from metro.utils.miscellaneous import mkdir, set_seed

from PIL import Image, ImageOps
from torchvision import transforms
from libzhifan.geometry import SimpleMesh, projection
import tqdm


transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])


def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """ 
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d


def project_hand(img, hand_verts, hand_faces, camera):
    """
    Args:
        camera: (3,)
    """
    focal_length = 1000 # constant in METRO
    res = img.shape[0]

    camera_t = torch.as_tensor(
        np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])
    )[None]

    image = torch.as_tensor(np.asarray(img))

    hand_mesh = SimpleMesh(hand_verts + camera_t.numpy(), hand_faces)  # note in METRO they apply to camera's T
    rend = projection.pytorch3d_perspective_projection(
        hand_mesh, cam_f=(focal_length,focal_length), cam_p=(res//2, res//2),
        image=image,
        in_ndc=False, coor_sys='nr')
    return rend


def run_inference(image_list, _metro_network, mano, renderer, mesh_sampler, output_dir, frames_root):
    """
    image_list: path to abs jpg files

    Outputs:
        - images will be saved to <output_dir>/rends
        - 3d predictions will be saved to <output_dir>/pred_3ds
        - 2d predictions will be saved to <output_dir>/pred_2ds
    """
    out_rend_dir = op.join(output_dir, 'rends')
    out_3d_dir = op.join(output_dir, 'pred_3ds')
    out_2d_dir = op.join(output_dir, 'pred_2ds')
    os.makedirs(out_rend_dir, exist_ok=True)
    os.makedirs(out_3d_dir, exist_ok=True)
    os.makedirs(out_2d_dir, exist_ok=True)

    # switch to evaluate mode
    _metro_network.eval()

    for img_name in tqdm.tqdm(image_list, total=len(image_list)):
        image_file = os.path.join(frames_root, img_name)
        side = 'left' if 'left' in img_name else 'right'
        # out_fpath = os.path.join(output_dir, os.path.basename(image_file))
        # if os.path.exists(out_fpath):
        #     continue

        img = Image.open(image_file)
        if side == 'left':
            img = ImageOps.mirror(img)
        img_tensor = transform(img)
        img_visual = transform_visualize(img)

        batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
        batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
        pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att = _metro_network(batch_imgs, mano, mesh_sampler)
        # obtain 3d joints from full mesh
        pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
        pred_3d_pelvis = pred_3d_joints_from_mesh[:,cfg.J_NAME.index('Wrist'),:]
        pred_3d_joints_from_mesh = pred_3d_joints_from_mesh - pred_3d_pelvis[:, None, :]
        pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

        # obtain 3d joints, which are regressed from the full mesh
        pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
        # obtain 2d joints, which are projected from 3d joints of mesh
        pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
        # pred_2d_coarse_vertices_from_mesh = orthographic_projection(pred_vertices_sub.contiguous(), pred_camera.contiguous())

        rend_img = project_hand(img=batch_visual_imgs[0].cpu().numpy().transpose(1, 2, 0),
                        hand_verts=pred_vertices[0].detach().cpu().numpy(),
                        hand_faces=renderer.faces,
                        camera=pred_camera.detach().cpu().numpy())
        if side == 'left':
            rend_img = rend_img[:, ::-1, :]

        """ Save rendered images """
        visual_imgs = rend_img
        out_fpath = os.path.join(out_rend_dir, os.path.basename(image_file))
        # print('save to ', out_fpath)
        cv2.imwrite(out_fpath, np.asarray(visual_imgs[:,:,::-1]*255))

        """ Save 3d predictions """ 
        out_3d_path = os.path.join(out_3d_dir, os.path.basename(image_file).replace('jpg', 'pth'))
        torch.save(pred_3d_joints_from_mesh, out_3d_path)

        """ Save 2d predictions """
        out_2d_path = os.path.join(out_2d_dir, os.path.basename(image_file).replace('jpg', 'pth'))
        torch.save(pred_2d_joints_from_mesh, out_2d_path)

    return


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--uid", type=str, default=None)
    parser.add_argument("--st", type=int, default=0)
    parser.add_argument("--ed", type=int, default=999999999999)
    parser.add_argument("--valid_mp4_segments", default='./epic_hor_data/valid_mp4_segments.json', type=str)
    # parser.add_argument("--image_file_or_path", default='./test_images/hand', type=str,
    #                     help="test data")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='metro/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False,
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False,
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str,
                        help="The Image Feature Dimension.")
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str,
                        help="The Image Feature Dimension.")
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88,
                        help="random seed for initialization.")


    args = parser.parse_args()
    return args

class DummyRenderer(object):
    def __init__(self):
        self.faces = None

def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    mkdir(args.output_dir)
    logger = setup_logger("METRO Inference", args.output_dir, 0)
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and MANO utils
    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.cuda()
    mesh_sampler = Mesh()
    # Renderer for visualization
    # renderer = Renderer(faces=mano_model.face)
    renderer = DummyRenderer()
    renderer.faces=mano_model.face

    # Load pretrained model
    logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))

    if args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _metro_network = torch.load(args.resume_checkpoint)
    else:
        # Build model from scratch, and load weights from state_dict.bin
        trans_encoder = []
        input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
        hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
        output_feat_dim = input_feat_dim[1:] + [3]
        # init three transformer encoders in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, METRO
            config = config_class.from_pretrained(args.model_name_or_path)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i]
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size*4)

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            logger.info("Init model from scratch.")
            trans_encoder.append(model)

        # init ImageNet pre-trained backbone model
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Transformers total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
        _metro_network = METRO_Network(args, config, backbone, trans_encoder)

        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _metro_network.load_state_dict(state_dict, strict=False)
        del state_dict

    # update configs to enable attention outputs
    setattr(_metro_network.trans_encoder[-1].config,'output_attentions', True)
    setattr(_metro_network.trans_encoder[-1].config,'output_hidden_states', True)
    _metro_network.trans_encoder[-1].bert.encoder.output_attentions = True
    _metro_network.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _metro_network.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_metro_network.trans_encoder[-1].config,'device', args.device)

    _metro_network.to(args.device)
    logger.info("Run inference")

    uid = args.uid
    assert uid is not None
    # df = pd.read_csv(f'./egopose_outputs/dets/{uid}.csv')
    frames_root = os.path.join('./egopose_outputs/handcrops/', uid)
    image_list = sorted(os.listdir(frames_root))
    image_list = image_list[1000:1001] # image_list = image_list[args.st:args.ed]
    output_dir = f'./egopose_outputs/results/{uid}/'
    run_inference(image_list, _metro_network, mano_model, renderer, mesh_sampler, output_dir=output_dir, frames_root=frames_root)

if __name__ == "__main__":
    args = parse_args()
    args.uid = '98f58f0f-53d6-4e41-bf41-d8d74ccbc37c'
    main(args)
