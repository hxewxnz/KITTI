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

import cv2
import numpy as np
import os
from os.path import dirname as ospd

from evaluation import BoxEvaluator
from evaluation import MaskEvaluator
from evaluation import configure_metadata
from util import t2n
from tqdm import tqdm 
from pytorch_grad_cam import *
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.metrics.cam_mult_image import ConfidenceChange 
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.image import scale_cam_image, get_target_width_height
import time 
import matplotlib.pyplot as plt
from PIL import Image
from util import unnormalize_image, save_cam_blend_from_ready_cams


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float32)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float32) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam

def reshape_transform(tensor, height=14, width=14):
    """ the first element represents the class token, and the rest represent the 14x14 patches in the image. """
    result = tensor[:, 1:, :].reshape(tensor.size(0), # BATCH x 197 x 192 -> BATCH x 196 x 14 x 14
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, log_folder=None,
                 model_structure='vanilla', unfreeze_layer='all', method='gradcam', **kwargs):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder
        self.model_structure = model_structure 
        self.unfreeze_layer = unfreeze_layer
        self.c1 = kwargs['c1'] 
        self.c2 = kwargs['c2'] 
        self.scoremap_path = kwargs['scoremap_path'] 
        self.architecture_type = kwargs['architecture_type']
        self.alpha = kwargs['alpha'] 
        self.architecture = kwargs['architecture'] 
        methods = {
            "gradcam": GradCAM,
            "hirescam": HiResCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "ablationcam": AblationCAM,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM,
            "eigengradcam": EigenGradCAM,
            "layercam": LayerCAM,
            "fullgrad": FullGrad,
            "fem": FEM,
            "gradcamelementwise": GradCAMElementWise,
            'kpcacam': KPCA_CAM,
            'shapleycam': ShapleyCAM,
            'finercam': FinerCAM
        }
        self.method = method
        self.cam_algorithm = methods[method] if method != 'cam' else None

        if self.model_structure == 'b2' and self.method != 'scorecam':
            self.truncate_weight = True
            print("[Branch 2] Truncate weight is set to True.")
        else: 
            self.truncate_weight = False

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = [round(x * cam_curve_interval, 3) for x in range(int(1 / cam_curve_interval) + 1)]

        self.evaluator = {"OpenImages": MaskEvaluator,
                        "CUB": BoxEvaluator,
                        "ILSVRC": BoxEvaluator,
                        'CARS' : BoxEvaluator,
                        'KITTI' : BoxEvaluator #수정
                        }[dataset_name](metadata=metadata,
                                        dataset_name=dataset_name,
                                        split=split,
                                        cam_threshold_list=cam_threshold_list,
                                        iou_threshold_list=iou_threshold_list,
                                        mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)

        # For deletion/insertion metric
        # self.percentiles = [i for i in range(0, 101)] # each 1% ~ 100%
        # self.relative_drop_percentiles = [5, 10]

    def compute_and_evaluate_gradcams(self, device, pred_correct_list, save_npy=True):
        
        print("Computing and evaluating gradcams.")
        
        # total_drop_scores = 0.0
        # total_increase_scores = 0.0
        # total_instances = 0
        # deletion_auc_lst = []
        # insertion_auc_lst = []
        deletion_confidence_dict = {}
        insertion_confidence_dict = {}

        for batch_idx, (images, labels, image_ids) in tqdm(enumerate(self.loader), desc='Computing GradCAMs', total=len(self.loader)):
            N = len(images)
            # P = len(self.percentiles)

            # deletion_matrix = np.zeros((N, P), dtype=np.float32)
            # insertion_matrix = np.zeros((N, P), dtype=np.float32)
            
            batch_pred_correct = pred_correct_list[batch_idx]
            image_size = images.shape[2:]
            images = images.to(device)
            # total_instances += N
               
            targets = [ClassifierOutputTarget(category) for category in labels] 

            """ 
            Define CAM algorithm and get CAMs.
            """
            if self.method == 'cam':
                pred_outputs = self.model(images, return_cam=True) # predicted logits from softmax head

                if self.model_structure == 'vanilla':
                    cams = t2n(self.model(images, labels, return_cam=True)['cams'])   
                    pred_based_cams = t2n(pred_outputs['cams']) # get cam from single head
                else: # 'b2'
                    cams = t2n(self.model(images, labels, return_cam=True)['cams2'])
                    pred_based_cams = t2n(pred_outputs['cams2']) # get cam from head2

                target_size = get_target_width_height(images)
                pred_based_cams = scale_cam_image(pred_based_cams, target_size) 

                pred_targets = pred_outputs['labels'].cpu().numpy() # get predicted labels from head1
                pred_targets = [ClassifierOutputTarget(category) for category in pred_targets]



            else:
                if self.architecture_type == 'vit':
                    if self.method == 'ablationcam':
                        cam_algorithm = self.cam_algorithm(model=self.model, 
                                                            target_layers=self.model.target_layers,
                                                            reshape_transform=reshape_transform,
                                                            ablation_layer=AblationLayerVit(),
                                                            device=device,)
                    else:
                        cam_algorithm = self.cam_algorithm(model=self.model, 
                                                            target_layers=self.model.target_layers,
                                                            reshape_transform=reshape_transform,
                                                           device=device,)
                
                else:
                    if self.method == 'finercam':
                        cam_algorithm = self.cam_algorithm(model=self.model, 
                                        target_layers=self.model.target_layers,
                                        device=device,
                                        alpha=self.alpha)

                    else:
                        cam_algorithm = self.cam_algorithm(model=self.model, 
                                        target_layers=self.model.target_layers,
                                        device=device,)
                        

                """ 
                Evaluate CAMs.
                """
                cam_algorithm.batch_size = len(images) # AblationCAM and ScoreCAM have batched implementations.You can override the internal batch size for faster computation.

                ###########
                target_layers = cam_algorithm.target_layers if hasattr(cam_algorithm, 'target_layers') else []
                for layer in target_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
                ###########

                if self.method == 'finercam':
                    cams = []
                    for i in range(images.shape[0]):
                        label = labels[i].item()  # scalar
                        image = images[i].unsqueeze(0)  # 1 x C x H x W
                        cam = cam_algorithm(
                            input_tensor=image,
                            truncate_weight=self.truncate_weight,
                            target_idx=label  
                        )
                        cams.append(cam[0])

                else:
                    cams = cam_algorithm(input_tensor=images, 
                                        targets=targets, 
                                        truncate_weight=self.truncate_weight,) 
                    #############
                    grads = cam_algorithm.activations_and_grads.gradients
                    assert grads is not None, \
                        f"Gradients are None for target layer(s): {[type(l).__name__ for l in target_layers]}"
                    #############
        
                # Debugging.. cams from prediction, not ground truth
                pred_based_cams, pred_targets = cam_algorithm(input_tensor=images, 
                                    targets=None, 
                                    truncate_weight=self.truncate_weight,) 

            # Drop in Confidence & Increase in confidence      
            # drop_scores, increase_scores, ori_score_0 = ConfidenceChange()(
            #     images, # batch_size x 3 x 224 x 224
            #     pred_based_cams, # batch_size x 224 x 224
            #     pred_targets, 
            #     self.model,
            #     return_ori_scores=True
            # )

            # total_drop_scores += np.sum(drop_scores)
            # total_increase_scores += np.sum(increase_scores)

  
            for idx, (cam, image_id, image) in enumerate(zip(cams, image_ids,images)):
                cam_resized = cv2.resize(cam, image_size, interpolation=cv2.INTER_CUBIC) 
                cam_normalized = normalize_scoremap(cam_resized) # 224 x 224

                """ 
                 # Deletion & Insertion Curve
                cam_flat = cam_normalized.flatten() # (50176,)
                sorted_indices = np.argsort(cam_flat)[::-1]  
                total_pixels = len(sorted_indices)

                target_confidences = []
                second_confidences = []
                removed_percentiles = np.linspace(0, 100, num=21)  # 0%, 5%, ..., 100%
                from pytorch_grad_cam.metrics.cam_mult_image import multiply_tensor_with_cam
                import torch 
                for p in removed_percentiles:
                    num_pixels_to_mask = int((p / 100.0) * total_pixels)
                    mask = np.ones(total_pixels, dtype=np.float32)
                    mask[sorted_indices[:num_pixels_to_mask]] = 0  # top pixels 제거

                    masked_cam = mask.reshape(cam_normalized.shape)
                    masked_input = multiply_tensor_with_cam(images[idx], torch.from_numpy(masked_cam).to(images.device))

                    with torch.no_grad():
                        output = self.model(masked_input.unsqueeze(0))
                        logits = output['logits2'] if 'logits2' in output else output['logits']
                        
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                    label = labels[idx].item()
                    top2_indices = probs.argsort()[::-1][:2]
                    target_conf = probs[label]
                    second_conf = probs[top2_indices[1]] if top2_indices[0] == label else probs[top2_indices[0]]

                    target_confidences.append(target_conf)
                    second_confidences.append(second_conf)
                # End of p loop
                # ex. self.scoremap_path : '/data/wsol/20250321/experiments/wsolevaluation/ILSVRC/resnet50_cam/scoremaps_gradcam/test'
                os.makedirs(os.path.dirname(os.path.join(self.scoremap_path, f"{image_id}_confidence_curve.npy")), exist_ok=True)
                np.save(os.path.join(self.scoremap_path, f"{image_id}_confidence_curve.npy"),
                        {"percentiles": removed_percentiles,
                        "target": target_confidences,
                        "second": second_confidences})
                """

                # if self.split in ('val', 'test'):            
                    # cam_path = os.path.join(os.path.dirname(self.scoremap_path), image_id) 
                    # if not os.path.exists(ospd(cam_path)):
                    #     os.makedirs(ospd(cam_path))
                    # np.save(os.path.join(cam_path), cam_normalized) 

                self.evaluator.accumulate(cam_normalized, image_id, batch_pred_correct[idx], image, self.method)

        return self.evaluator.compute() #, total_drop_scores, total_increase_scores, deletion_auc_lst, insertion_auc_lst, total_instances
    
