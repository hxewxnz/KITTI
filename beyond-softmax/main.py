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
""" 
You need to choose the target layer to compute the CAM for. Some common choices are:

FasterRCNN: model.backbone
Resnet18 and 50: model.layer4[-1]
VGG, densenet161 and mobilenet: model.features[-1]
mnasnet1_0: model.layers[-1]
ViT: model.blocks[-1].norm1
SwinT: model.layers[-1].blocks[-1].norm1

"""
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim

from config import get_configs
from data_loaders import get_data_loader
from inference import CAMComputer
from util import string_contains_any, plot_confidence_curves
import wsol
import wsol.method
import wsol_cam
import wsol_cam.method
import util
from tqdm import tqdm 
import wandb 
import time 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

global SANITY_CHECK
SANITY_CHECK = False
def set_bn_to_eval(m):
    global SANITY_CHECK
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if not SANITY_CHECK: 
            print(f'[Debug][main.py] Apply eval mode to BatchNorm')
            SANITY_CHECK = True
        m.eval()

class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.value_per_epoch = [] \
            if split == 'val' else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'top1_cls', 'maxboxacc_v2', 'top1_loc', 'gt_loc',]
    _BEST_CRITERION_METRIC = 'top1_loc' # 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "KITTI": 9, #수정
        "ILSVRC": 1000,
        "OpenImages": 100,
        'CARS': 196 
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
        'vit': ['patch_embed.', 'blocks.', 'norm'], 
    }
    _IMAGE_MEAN_MAPPING = {
        'vgg16': [0.485, 0.456, 0.406],
        'resnet50': [0.485, 0.456, 0.406],
        'inception_v3': [0.485, 0.456, 0.406],
        'vit': [0.5, 0.5, 0.5], 
    }
    _IMAGE_STD_MAPPING = {
        'vgg16': [0.229, 0.224, 0.225],
        'resnet50': [0.229, 0.224, 0.225],
        'inception_v3': [0.229, 0.224, 0.225],
        'vit': [0.5, 0.5, 0.5], 
    }
    def __init__(self, args, split='train'):
        
        self.args = args #get_configs()
        self.dataset_name = self.args.dataset_name
        set_random_seed(self.args.seed)
        print(self.args)
        print(f'Seed: {self.args.seed}')
        #self.device_ids = list(map(int, self.args.gpus.split(','))) 
        self.device_ids = list(range(torch.cuda.device_count()))
        device = f'cuda:{self.device_ids[0]}' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.model_structure = self.args.model_structure 
        print('⭐ Model structure: {}'.format(self.model_structure))

        if self.dataset_name == 'OpenImages':
            self.performance_meters  = {
                split: {
                    metric: PerformanceMeter(split,
                                            higher_is_better=False
                                            if metric == 'loss' else True)
                    for metric in ['loss', 'PxAP', 'top1_cls',]
                }
                for split in self._SPLITS
        }

        else:
            self.performance_meters = self._set_performance_meters() 

        self.reporter = self.args.reporter
        self.debug = args.debug
        self.wandb = args.wandb
    
        self.model = self._set_model()
        if split == 'train':
            self.model = util.DataParallel(self.model, device_ids=self.device_ids)

        if split == 'test' or args.unfreeze_layer != 'all':
            self.model.apply(set_bn_to_eval)
        
        self.model.to(self.device) 


        from criterion import CrossEntropyLoss, BinaryCrossEntropyLoss
        CRITERIONS = {
            'cross_entropy': CrossEntropyLoss, # softmax
            'binary_cross_entropy': BinaryCrossEntropyLoss # sigmoid
        }
        self.criterion_1 = CRITERIONS[self.args.c1](num_classes=self._NUM_CLASSES_MAPPING[self.args.dataset_name])
        self.criterion_2 = CRITERIONS[self.args.c2](num_classes=self._NUM_CLASSES_MAPPING[self.args.dataset_name]) 

        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = self._set_optimizer()
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            mean=self._IMAGE_MEAN_MAPPING[self.args.architecture],
            std=self._IMAGE_STD_MAPPING[self.args.architecture],
            )
        
        self.percentiles = [i for i in range(0, 101)] # each 1% ~ 100%

    def _set_performance_meters(self):
        self._EVAL_METRICS += ['maxboxacc_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("[Debug][main.py] Loading model {}".format(self.args.architecture))

        common_kwargs = dict(
            dataset_name=self.args.dataset_name,
            architecture_type=self.args.architecture_type,
            pretrained=self.args.pretrained,
            pretrained_path=self.args.pretrained_path,
            num_classes=num_classes,
            large_feature_map=self.args.large_feature_map,
            adl_drop_rate=self.args.adl_drop_rate,
            adl_drop_threshold=self.args.adl_threshold,
            acol_drop_threshold=self.args.acol_threshold,

            last_layer=self.args.last_layer,  
            unfreeze_layer=self.args.unfreeze_layer,
            model_structure=self.args.model_structure,
            init_weights=self.args.init_weight,
            debug=self.args.debug,
            ft_ckpt=self.args.ft_ckpt,
        )

        model_class = wsol_cam if self.args.method == 'cam' else wsol
        model = model_class.__dict__[self.args.architecture](**common_kwargs)

        return model

    def _set_optimizer(self):
        param_features = []
        param_classifiers = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        for name, parameter in self.model.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.args.architecture)):
                if self.args.architecture in ('vgg16', 'inception_v3', 'vit'):
                    param_features.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_classifiers.append(parameter)
            else:
                if self.args.architecture in ('vgg16', 'inception_v3', 'vit'):
                    param_classifiers.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_features.append(parameter)

        optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': self.args.lr},
            {'params': param_classifiers,
             'lr': self.args.lr * self.args.lr_classifier_ratio}],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True)
        return optimizer

    def _get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def _wsol_training(self, images, target, epoch, batch_idx, tot_batch):
        if (self.args.wsol_method == 'cutmix' and
                self.args.cutmix_prob > np.random.rand(1) and
                self.args.cutmix_beta > 0):
            images, target_a, target_b, lam = wsol.method.cutmix(
                images, target, self.args.cutmix_beta)
            output_dict = self.model(images)
            logits = output_dict['logits']
            loss = (self.cross_entropy_loss(logits, target_a) * lam +
                    self.cross_entropy_loss(logits, target_b) * (1. - lam))
            return logits, loss

        if self.args.wsol_method == 'has':
            images = wsol.method.has(images, self.args.has_grid_size,self.args.has_drop_rate)

        output_dict = self.model(images, target)
        if self.model_structure == 'b2':
            logits = output_dict['logits'] 
            logits2 = output_dict['logits2'] 
        else:
            logits = output_dict['logits']

        if self.args.wsol_method in ('acol', 'spg'):
            loss = wsol.method.__dict__[self.args.wsol_method].get_loss(
                output_dict, target, spg_thresholds=self.args.spg_thresholds)
        else:
            if self.model_structure == 'b2':
                loss1 = self.criterion_1(logits, target)
                loss2 = self.criterion_2(logits2, target) 
                loss = loss1 + loss2
                print(f'[Epoch {epoch}][Batch {batch_idx}/{tot_batch}] Loss: {loss:.4f} (Loss1: {loss1:.4f}, Loss2: {loss2:.4f})')
            elif self.model_structure == 'b1':
                loss1 = self.criterion_1(logits, target) 
                loss2 = self.criterion_2(logits, target) 
                loss = loss1 + loss2 
                print(f'[Epoch {epoch}][Batch {batch_idx}/{tot_batch}] Loss: {loss:.4f} (Loss1: {loss1:.4f}, Loss2: {loss2:.4f})') 
            else:
                # vanilla
                loss = self.criterion_1(logits, target)
                print(f'[Epoch {epoch}][Batch {batch_idx}/{tot_batch}] Loss: {loss:.4f}')

        return logits, loss

    def train(self, split, epoch):
        if not isinstance(self.model, torch.nn.DataParallel):
            print("[Debug][train] Wrapping model with DataParallel again.")
            self.model = util.DataParallel(self.model, device_ids=self.device_ids)

        
        if self.args.unfreeze_layer != 'all':
            print(f'[Debug][train] Re-freezing layers at epoch {epoch}')
            self.model.module.freeze_layers()

        print(f'==> Model structure: {self.model_structure} <==')
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
        print("===========================================================")
        
        self.model.train()

        loader = self.loaders[split]

        total_loss = 0.0
        num_correct = 0
        num_images = 0
        tot_batch = len(loader)
        for batch_idx, (images, target, _) in enumerate(loader):
            images = images.to(self.device)
            target = target.to(self.device)

            if batch_idx % int(len(loader) / 10) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))

            logits, loss = self._wsol_training(images, target, epoch, batch_idx + 1, tot_batch)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            num_correct += (pred == target).sum().item()
            num_images += images.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            torch.cuda.synchronize() 
            self.optimizer.step()

        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) * 100

        self.performance_meters[split]['loss'].update(loss_average)

        wandb.log({
            'learning_rate': self._get_lr()[0],
        })
        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def print_performances(self, epoch):
        if self.dataset_name == 'OpenImages':
            _eval_metrics = ['PxAP', 'top1_cls', ]
        else:
            _eval_metrics = self._EVAL_METRICS
        for split in self._SPLITS:
            for metric in _eval_metrics:
                current_performance = self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    print("Split {}, metric {}, current value: {}".format(split, metric, current_performance))
                    if split == 'test':
                        if self.wandb == 'on':
                            wandb.log(
                                {f'{metric}': current_performance}, #step=epoch
                            )
                    if split != 'test':
                        print("Split {}, metric {}, best value: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_value))
                        print("Split {}, metric {}, best epoch: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_epoch)) 
                        
    def save_performances(self, epoch:int=-1):
        if epoch != -1:
            pickle_fname = 'epoch_{}_'.format(epoch) 
        else:
            pickle_fname = ''
        log_path = os.path.join(self.args.log_folder, f'{pickle_fname}performance_log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.performance_meters, f)

    def _compute_accuracy(self, loader,):
        num_correct = 0
        num_images = 0
        correct_list = [] 
        num_correct_2 = 0 

        for i, (images, targets, image_ids) in tqdm(enumerate(loader), desc="Calculating accuracy", total=len(loader)):
            images = images.to(self.device)
            targets = targets.to(self.device)
            output_dict = self.model(images)
            pred = output_dict['logits'].argmax(dim=1)

            if self.model_structure == 'b2':
                pred2 = output_dict['logits2'].argmax(dim=1)
                num_correct_2 += (pred2 == targets).sum().item()

            batch_correct = (pred == targets).int()
            correct_list.append(batch_correct.cpu().numpy())

            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100
        classification_acc_2 = num_correct_2 / float(num_images) * 100
        return classification_acc, correct_list, classification_acc_2

    
    def evaluate(self, epoch='', split='test', iou_threshold=50, loc_threshold=0.2):
        if epoch:
            print("Evaluate epoch {}, split {}".format(epoch, split)) 

        if isinstance(self.model, torch.nn.DataParallel):
            print("[Debug][evaluate] Unwrapping DataParallel to get original model")
            self.model = self.model.module

        def set_bn_to_eval(m): 
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.model.apply(set_bn_to_eval)
        self.model.eval()

        accuracy, pred_correct_list, accuracy_2 = self._compute_accuracy(loader=self.loaders['test'])
        print(accuracy) 
        print('-----------------------------')
        self.performance_meters[split]['top1_cls'].update(accuracy)

    
        cam_computer = CAMComputer(
            model=self.model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset_name,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            log_folder=self.args.log_folder,
            model_structure=self.model_structure,
            c1=self.args.c1,
            c2=self.args.c2,
            scoremap_path=self.args.scoremap_paths[split],
            method=self.args.method,
            architecture_type = self.args.architecture,
            alpha=self.args.alpha,
            architecture=self.args.architecture
        )
        cam_threshold_list = [round(x * self.args.cam_curve_interval, 3) for x in range(int(1 / self.args.cam_curve_interval) + 1)]
        
        if self.dataset_name == 'OpenImages':
            performance  =  cam_computer.compute_and_evaluate_gradcams(self.device, pred_correct_list)
            loc_score = np.average(performance)

            # drop_score = (total_drop_scores / total_instances) * 100.0
            # increase_score = (total_increase_scores / total_instances) * 100.0
        
            self.performance_meters[split]['PxAP'].update(loc_score) 
            # self.performance_meters[split]['avg_drop'].update(drop_score)
            # self.performance_meters[split]['avg_increase'].update(increase_score)
            # self.performance_meters[split]['deletion_auc'].update(deletion_auc)
            # self.performance_meters[split]['insertion_auc'].update(insertion_auc)
            return epoch

        cam_performance, gt_loc_performance, top1_loc_performance = cam_computer.compute_and_evaluate_gradcams(self.device, pred_correct_list)

        # drop_score = (total_drop_scores / total_instances) * 100.0
        # increase_score = (total_increase_scores / total_instances) * 100.0
        
        # deletion_auc  = np.mean(deletion_auc_lst)
        # insertion_auc = np.mean(insertion_auc_lst)


        if self.args.multi_iou_eval or self.args.dataset_name == 'OpenImages':
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(iou_threshold)]

        gt_loc_score = gt_loc_performance[self.args.iou_threshold_list.index(iou_threshold)][cam_threshold_list.index(loc_threshold)] 
        top1_loc_score = top1_loc_performance[self.args.iou_threshold_list.index(iou_threshold)][cam_threshold_list.index(loc_threshold)] 
        
        
        self.performance_meters[split]['maxboxacc_v2'].update(loc_score)
        self.performance_meters[split]['gt_loc'].update(gt_loc_score) 
        self.performance_meters[split]['top1_loc'].update(top1_loc_score)

        if self.args.dataset_name in ('CUB', 'ILSVRC', 'CARS', "KITTI"): #수정
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split][
                    'maxboxacc_IOU_{}'.format(IOU_THRESHOLD)].update(
                    cam_performance[idx])    
                
        # self.performance_meters[split]['avg_drop'].update(drop_score)
        # self.performance_meters[split]['avg_increase'].update(increase_score)
        # self.performance_meters[split]['deletion_auc'].update(deletion_auc)
        # self.performance_meters[split]['insertion_auc'].update(insertion_auc)
        # self.performance_meters[split]['proportion'].update(proportion_performance)

        return epoch #, drop_score, increase_score, proportion_performance
    
    def _torch_save_model(self, filename, epoch):
        torch.save({'architecture': self.args.architecture,
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.log_folder, filename))

    def save_checkpoint(self, epoch,):
        if self.dataset_name == 'OpenImages':
            _best_criterion_metric = 'PxAP' 
        else:
            _best_criterion_metric = self._BEST_CRITERION_METRIC


        # if (self.performance_meters[split][_best_criterion_metric].best_epoch) == epoch:
        #     self._torch_save_model(self._CHECKPOINT_NAME_TEMPLATE.format('best'), epoch)


        if self.args.epochs == epoch:
            self._torch_save_model(self._CHECKPOINT_NAME_TEMPLATE.format('last'), epoch)

        elif epoch == 'highest':
            self._torch_save_model(self._CHECKPOINT_NAME_TEMPLATE.format('highest'), epoch)

        else: 
            self._torch_save_model(self._CHECKPOINT_NAME_TEMPLATE.format(epoch), epoch)


            
    def report_train(self, train_performance, epoch, split='train'):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        reporter_instance.add(
            key='{split}/classification'.format(split=split),
            val=train_performance['classification_acc'])
        reporter_instance.add(
            key='{split}/loss'.format(split=split),
            val=train_performance['loss'])
        reporter_instance.write()

        if self.wandb == 'on':
            wandb.log(
                {'classification_acc': train_performance['classification_acc'],
                'loss': train_performance['loss']
                },
                #
                # step = epoch 
            )

    def report(self, epoch, split):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        if self.dataset_name == 'OpenImages':
            _eval_metrics = ['PxAP', ]
        else:
            _eval_metrics = self._EVAL_METRICS
        for metric in _eval_metrics:
            reporter_instance.add(
                key='{split}/{metric}'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].current_value)
            reporter_instance.add(
                key='{split}/{metric}_best'.format(split=split, metric=metric),
                val=self.performance_meters[split][metric].best_value)
        reporter_instance.write() 


    def adjust_learning_rate(self, epoch):
        if epoch != 0 and epoch % self.args.lr_decay_frequency == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1

    def load_checkpoint(self, checkpoint_type):
        print("Loading {} checkpoint.".format(checkpoint_type))
        try:
            checkpoint_path = os.path.join(
                self.args.log_folder, 
                self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
            print("==> Checkpoint path: {}".format(checkpoint_path))
        except:
            checkpoint_path = checkpoint_type 

        # Wait for the checkpoint to appear if it does not exist
        while not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found. Waiting for the next checkpoint...")
            time.sleep(30)  # Wait for 10 seconds before checking again

        if not os.path.exists(checkpoint_path):
            checkpoint_type = 'best'
            checkpoint_path = os.path.join(
                self.args.log_folder, 
                self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
            
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            

            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#            state_dict = {k.replace('module.module.', 'module.features.'): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=True)
#            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))

class SimpleForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['logits']

def main():
    import torch
    torch.cuda.init()
    args = get_configs()

    if args.wandb == 'on':
        print(f'===> Wandb project: {args.project}, name: {args.experiment_name}')
        print(f'===> method : {args.method}')
        config_dict = {
            'epochs': args.epochs,
            'c1': args.c1,
            'c2': args.c2,
            'data': args.dataset_name,
            'batch_size': args.batch_size,
            'iou_lst': args.iou_threshold_list,
            'lr': args.lr,
            'lr_decay_freq': args.lr_decay_frequency,
            'model_structure': args.model_structure,
            'last_layer': args.last_layer,
            'unfreeze_layer': args.unfreeze_layer,
            'seed': args.seed,
            'method': args.method,
            'loc_threshold': args.loc_threshold,
        }
        tag_lst = [args.model_structure, args.experiment_name, args.dataset_name] 

        if args.wandb_name is not None:
            wandb_name = args.wandb_name 
        else:
            wandb_name = args.experiment_name
        wandb.init(
                project=args.project, 
                name=wandb_name,
                dir=args.wandb_dir,
                config=config_dict,
                tags=tag_lst
                )


    if not args.eval_only:
        trainer = Trainer(args=args, split='train') 
        #trainer.load_checkpoint(checkpoint_type='current', epoch=7)

        if trainer.args.epochs == 0:
            #trainer.save_checkpoint(0, split='train')
            trainer.save_checkpoint(0)

        for epoch in range(trainer.args.epochs):
            print("===========================================================")
            print("Start epoch {} ...".format(epoch + 1))
            trainer.adjust_learning_rate(epoch + 1)
            train_performance = trainer.train(split='train', epoch=epoch + 1)
            
            trainer.report_train(train_performance, epoch + 1, split='train')
            #trainer.save_checkpoint(epoch + 1, split='train')
            trainer.save_checkpoint(epoch + 1)
            print("Epoch {} done.".format(epoch + 1))

    print("===========================================================")
    print("Final epoch evaluation on test set ...")

    print(f'Method: {args.method}')
    print(f' Localization threshold: {args.loc_threshold}')

    if not args.train_only:
        if args.eval_checkpoint_type == 'current': 
            print("🚨Start epoch 0 ...")
            trainer = Trainer(args=args, split='test')
            epoch = trainer.evaluate(epoch=0, split='test', loc_threshold=args.loc_threshold) 
            trainer.print_performances(epoch=0)
            trainer.report(epoch=0, split='test')
            trainer.save_performances(epoch=0)
            print("--> Epoch 0 done.")

            trainer = Trainer(args=args, split='test')
            total_epochs = args.epochs 
            for current_epoch in range(1, total_epochs + 1):
                trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type, epoch=current_epoch)
                epoch = trainer.evaluate(current_epoch, split='test',loc_threshold=args.loc_threshold) 
                trainer.print_performances(epoch=epoch)
                trainer.report(current_epoch, split='test')
                trainer.save_performances()
                print("===========================================================")
        else:
            trainer = Trainer(args=args, split='test')
            trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)
            epoch = trainer.evaluate(trainer.args.epochs, split='test', loc_threshold=args.loc_threshold) 
            trainer.print_performances(epoch=epoch)
            trainer.report(trainer.args.epochs, split='test')
            trainer.save_performances()

if __name__ == '__main__':
    main()
