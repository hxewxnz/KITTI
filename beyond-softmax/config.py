"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import munch
import importlib
import os
import shutil
import warnings
from datetime import datetime
from util import Logger

_DATASET_NAMES = ['CUB', 'ILSVRC', 'OpenImages','CARS', 'KITTI']
_ARCHITECTURE_NAMES = ('vgg16', 'resnet50', 'inception_v3', 'vit')
_METHOD_NAMES = ('cam', 'adl', 'acol', 'spg', 'has', 'cutmix')
_SPLITS = ('train', 'val', 'test')


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def box_v2_metric(args):
    if args.box_v2_metric:
        args.multi_contour_eval = True
        args.multi_iou_eval = True
    else:
        args.multi_contour_eval = False
        args.multi_iou_eval = False
        warnings.warn("MaxBoxAcc metric is deprecated.")
        warnings.warn("Use MaxBoxAccV2 by setting args.box_v2_metric to True.")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_architecture_type(wsol_method):
    if wsol_method in ('has', 'cutmix'):
        architecture_type = 'cam'
    else:
        architecture_type = wsol_method
    return architecture_type


def configure_data_paths(args):
    train = val = test = os.path.join(args.data_root, args.dataset_name)
    data_paths = mch(train=train, val=val, test=test)
    return data_paths


def configure_mask_root(args):
    mask_root = os.path.join(args.mask_root, 'OpenImages')
    return mask_root


def configure_scoremap_output_paths(args):
    scoremaps_root = os.path.join(args.log_folder, 'scoremaps_gradcam' )
    scoremaps = mch()
    for split in ('train', 'val', 'test'):
        scoremaps[split] = os.path.join(scoremaps_root, split)
        if not os.path.isdir(scoremaps[split]):
            os.makedirs(scoremaps[split])
    return scoremaps

def configure_reporter(args):
    if args.t1 and not args.t2:
        folder_name = 'reports_cmp_sm' 
    if args.t2 and not args.t1:
        folder_name = 'reports_cmp'
    elif args.t1 and args.t2:
        folder_name = 'reports_cmp_all'
    else:
        folder_name = 'reports'

    folder_name = folder_name + f'_{args.method}'

    reporter = importlib.import_module('util').Reporter
    reporter_log_root = os.path.join(args.log_folder, folder_name)
    if not os.path.isdir(reporter_log_root):
        os.makedirs(reporter_log_root)
    print(f'[Debug][config.py] reporter_log_root: {reporter_log_root}')
    return reporter, reporter_log_root


def configure_log_folder(args):
    log_folder = os.path.join(args.root, args.experiment_name)
    print(f'[Debug][config.py] log_folder: {log_folder}')
    
    if not args.eval_only:
        if os.path.isdir(log_folder):
            if args.override_cache:
                shutil.rmtree(log_folder, ignore_errors=True)
            else:
                raise RuntimeError("Experiment with the same name exists: {}".format(log_folder))
        os.makedirs(log_folder)

    return log_folder


def configure_log(args):
    # log_file_name = os.path.join(args.log_folder, 'log.log')
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
    log_file_name = os.path.join(args.log_folder, f"{timestamp}.log")
    Logger(log_file_name)

def configure_pretrained_path(args):
    pretrained_path = None
    return pretrained_path


def check_dependency(args):
    if args.dataset_name == 'CUB':
        if args.num_val_sample_per_class >= 6:
            raise ValueError("num-val-sample must be <= 5 for CUB.")
    if args.dataset_name == 'OpenImages':
        if args.num_val_sample_per_class >= 26:
            raise ValueError("num-val-sample must be <= 25 for OpenImages.")


def get_configs():
    parser = argparse.ArgumentParser()

    # Util
    parser.add_argument('--seed', type=int)
    parser.add_argument('--experiment_name', type=str, default='test_case')
    parser.add_argument('--override_cache', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')

    # Data
    parser.add_argument('--dataset_name', type=str, default='CUB',
                        choices=_DATASET_NAMES)
    parser.add_argument('--data_root', metavar='/PATH/TO/DATASET',
                        default='/content/drive/MyDrive/beyond-softmax-master/dataset/', 
                        help='path to dataset images')
    parser.add_argument('--metadata_root', type=str, default='metadata/')
    parser.add_argument('--mask_root', metavar='/PATH/TO/MASKS',
                        default='dataset/',
                        help='path to masks')
    parser.add_argument('--proxy_training_set', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')
    parser.add_argument('--num_val_sample_per_class', type=int, default=0,
                        help='Number of full_supervision validation sample per '
                             'class. 0 means "use all available samples".')

    # Setting
    parser.add_argument('--architecture', default='resnet18',
                        choices=_ARCHITECTURE_NAMES,
                        help='model architecture: ' +
                             ' | '.join(_ARCHITECTURE_NAMES) +
                             ' (default: resnet18)')
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--pretrained', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Use pre_trained model.')
    parser.add_argument('--cam_curve_interval', type=float, default=.001,
                        help='CAM curve interval')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='input resize size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='input crop size')
    parser.add_argument('--multi_contour_eval', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--multi_iou_eval', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--iou_threshold_list', nargs='+',
                        type=int, default=[30, 50, 70])
    parser.add_argument('--eval_checkpoint_type', type=str, default='last',)
                   #     choices=('best', 'last', 'current', 'highest','epoch_12', '3_tcon2', '2_tcon2','12_tcon2', '5_tcon2', ))
    parser.add_argument('--box_v2_metric', type=str2bool, nargs='?',
                        const=True, default=True)

    # Common hyperparameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Mini-batch size (default: 256), this is the total'
                             'batch size of all GPUs on the current node when'
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--lr_decay_frequency', type=int, default=30,
                        help='How frequently do we decay the learning rate?')
    parser.add_argument('--lr_classifier_ratio', type=float, default=10,
                        help='Multiplicative factor on the classifier layer.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--large_feature_map', type=str2bool, nargs='?',
                        const=True, default=False)

    # Method-specific hyperparameters
    parser.add_argument('--wsol_method', type=str, default='base',
                        choices=_METHOD_NAMES)
    parser.add_argument('--has_grid_size', type=int, default=4)
    parser.add_argument('--has_drop_rate', type=float, default=0.5)
    parser.add_argument('--acol_threshold', type=float, default=0.7)
    parser.add_argument('--spg_threshold_1h', type=float, default=0.7,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_1l', type=float, default=0.01,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2h', type=float, default=0.5,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2l', type=float, default=0.05,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3h', type=float, default=0.7,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3l', type=float, default=0.1,
                        help='SPG threshold')
    parser.add_argument('--adl_drop_rate', type=float, default=0.75,
                        help='ADL dropout rate')
    parser.add_argument('--adl_threshold', type=float, default=0.9,
                        help='ADL gamma, threshold ratio '
                             'to maximum value of attention map')
    parser.add_argument('--cutmix_beta', type=float, default=1.0,
                        help='CutMix beta')
    parser.add_argument('--cutmix_prob', type=float, default=1.0,
                        help='CutMix Mixing Probability')
    parser.add_argument('--gpus', type=str, default="5",)
    parser.add_argument('--root', type=str, default="/data/wsol/20250321/experiments/wsolevaluation/ILSVRC",)
    parser.add_argument('--wandb', default='on') 
    parser.add_argument('--project', default='vgg16-imagenet')
    parser.add_argument('--wandb_dir', default="/data/wsol/20250321/experiments/wsolevaluation/", type=str,)
    parser.add_argument('--last_layer', default='fc', help='[fc, conv]')
    parser.add_argument('--unfreeze_layer', default='all', help='[all, fc, fc2, None]')
    parser.add_argument('--model_structure', default='vanilla', help='[vanilla, b1, b2]')
    parser.add_argument('--eval_only', default=False, action='store_true', help='Evaluation only mode')    
    parser.add_argument('--train_only',default=False, action='store_true',)
    parser.add_argument('--init_weight', default=True, help='[True, False]') 
    parser.add_argument('--debug', default=True, action='store_true', help="Turn on debug mode")
    parser.add_argument('--c1', default='cross_entropy', help='[cross_entropy, binary_cross_entropy]')
    parser.add_argument('--c2', default='binary_cross_entropy', help='[cross_entropy, binary_cross_entropy]')
    parser.add_argument('--t1', default=False, action='store_true', help='truncate head1 mode')
    parser.add_argument('--t2', default=True, action='store_true', help='truncate head2 mode')
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--ft_ckpt', default=None, type=str, help='Fine-tuning checkpoint path')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'fem', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam', 'shapleycam',
                            'finercam', 'cam' 
                        ],
                        help='CAM method')
    parser.add_argument('--loc_threshold', type=float, default=0.2,) #Add
    parser.add_argument('--alpha', type=float, default=0.6) 
    def parse_methods(methods_str):
        return methods_str.split(',')

    parser.add_argument('--methods', type=parse_methods, default=None, help='Comma-separated list of methods, e.g., "gradcam,gradcam++"')
    args = parser.parse_args()

    check_dependency(args)
    args.log_folder = configure_log_folder(args)
    configure_log(args)
    box_v2_metric(args)

    args.architecture_type = get_architecture_type(args.wsol_method)
    args.data_paths = configure_data_paths(args)
    args.metadata_root = os.path.join(args.metadata_root, args.dataset_name)
    args.mask_root = configure_mask_root(args)
    args.scoremap_paths = configure_scoremap_output_paths(args)
    args.reporter, args.reporter_log_root = configure_reporter(args)
    args.pretrained_path = configure_pretrained_path(args)
    args.spg_thresholds = ((args.spg_threshold_1h, args.spg_threshold_1l),
                           (args.spg_threshold_2h, args.spg_threshold_2l),
                           (args.spg_threshold_3h, args.spg_threshold_3l))

    print(f'[Debug][config.py] Root of exps is {args.root}')
    def mapping_name():
        if args.model_structure == 'vanilla' :
            if args.c1 == 'binary_cross_entropy':
                return args.architecture + f'_{args.wsol_method}_sig'
            else:
                return args.architecture 
        elif args.model_structure == 'b1' :
            return args.architecture + f'_{args.wsol_method}_b1' 
        elif args.model_structure == 'b2' :
            if args.unfreeze_layer == 'all':
                return args.architecture + f'_{args.wsol_method}_b2' 
            elif args.unfreeze_layer == 'fc':
                return args.architecture + f'_{args.wsol_method}_b2_fzf2'
            elif args.unfreeze_layer == 'fc2':
                return args.architecture + f'_{args.wsol_method}_b2_fzf1' 
    
    assert args.model_structure in ['vanilla', 'b1', 'b2'], f"Invalid model_structure: {args.model_structure}"
    assert args.last_layer in ['fc', 'conv'], f"Invalid last_layer: {args.last_layer}" 
    print(f'[Debug][config.py] MODEL: {mapping_name()} (last_layer: {args.last_layer})')

    if args.seed is None: 
        args.seed = 42
    return args
