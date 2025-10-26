python main.py `
  --dataset_name ILSVRC `
  --architecture vgg16 `
  --wsol_method cam `
  --method cam `
  --experiment_name vgg16_cam_large `
  --wandb_name vgg16_cam_large `
  --root "C:/Users/User/Desktop/grad/experiment/imagenet" `
  --epoch 10 `
  --batch_size 32 `
  --lr_decay_frequency 3 `
  --workers 4 `
  --gpus 0 `
  --lr 0.0001 `
  --weight_decay 5.00E-04 `
  --model_structure vanilla `
  --size_group large_objects `
  --unfreeze_layer all `
  --project vgg16-0 `
  --metadata_root "C:/Users/User/Desktop/grad/metadata"

python main.py `
  --dataset_name ILSVRC `
  --architecture vgg16 `
  --wsol_method cam `
  --method cam `
  --experiment_name vgg16_cam_large `
  --wandb_name vgg16_cam_large `
  --root "C:/Users/User/Desktop/grad/experiment/imagenet" `
  --epoch 10 `
  --batch_size 32 `
  --lr_decay_frequency 3 `
  --workers 4 `
  --gpus 0 `
  --lr 0.003 `
  --weight_decay 5.00E-04 `  
  --model_structure b2 `
  --size_group large_objects `
  --ft_ckpt C:/Users/User/Desktop/grad/experiment/imagenet/vgg16_cam_large/last_checkpoint.pth.tar `
  --unfreeze_layer fc2 `
  --project vgg16-0 `
  --metadata_root "C:/Users/User/Desktop/grad/metadata"

  ####################################################

python main.py \
  --dataset_name ILSVRC \
  --architecture vgg16 \
  --wsol_method cam \
  --method cam \
  --experiment_name vgg16_cam_ours \
  --wandb_name vgg16_cam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/ \
  --large_feature_map FALSE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 3 \
  --workers 4 \
  --gpus 1 \
  --lr 0.003 \
  --weight_decay 5.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/vgg16/vgg16_cam/last_checkpoint.pth.tar \
  --unfreeze_layer fc2 \
  --project vgg16-0 


# 1. Imagenet
# 1-1. vgg16_cam
python main.py \
  --dataset_name ILSVRC \
  --architecture vgg16 \
  --wsol_method cam \
  --method cam \
  --experiment_name vgg16_cam \
  --wandb_name vgg16_cam \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/ \
  --large_feature_map FALSE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 3 \
  --workers 4 \
  --gpus 1 \
  --lr 0.00001654502 \
  --weight_decay 5.00E-04 \
  --model_structure vanilla \
  --size_group large \
  --project vgg16-0 

# 1-2. vgg16_cam + ours 
python main.py \
  --dataset_name ILSVRC \
  --architecture vgg16 \
  --wsol_method cam \
  --method cam \
  --experiment_name vgg16_cam_ours \
  --wandb_name vgg16_cam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/ \
  --large_feature_map FALSE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 3 \
  --workers 4 \
  --gpus 1 \
  --lr 0.003 \
  --weight_decay 5.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/vgg16/vgg16_cam/last_checkpoint.pth.tar \
  --unfreeze_layer fc2 \
  --project vgg16-0 

# 1-3. vgg16_gradcam (no training)

# 1-4. vgg16_gradcam + ours
python main.py \
  --dataset_name ILSVRC \
  --architecture vgg16 \
  --wsol_method cam \
  --method gradcam \
  --experiment_name vgg16_gradcam_ours \
  --wandb_name vgg16_gradcam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/ \
  --large_feature_map FALSE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 3 \
  --workers 4 \
  --gpus 1 \
  --lr 0.0001 \
  --weight_decay 5.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/vgg16/vgg16_gradcam/last_checkpoint.pth.tar \
  --unfreeze_layer classifier_2 \
  --project vgg16-0 

# 2-1. resnet50_cam
python main.py \
  --dataset_name ILSVRC \
  --architecture resnet50 \
  --wsol_method cam \
  --method cam \
  --experiment_name resnet50_cam \
  --wandb_name resnet50_cam \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/ \
  --large_feature_map TRUE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 3 \
  --workers 4 \
  --gpus 1 \
  --lr 0.00003108411 \
  --weight_decay 1.00E-04 \
  --model_structure vanilla \
  --project resnet50-0 

# 2-2. resnet50_cam + ours
python main.py \
  --dataset_name ILSVRC \
  --architecture resnet50 \
  --wsol_method cam \
  --method cam \
  --experiment_name resnet50_cam_ours \
  --wandb_name resnet50_cam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/ \
  --large_feature_map TRUE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 3 \
  --workers 4 \
  --gpus 1 \
  --lr 0.0001 \
  --weight_decay 1.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/resnet50/resnet50_cam/last_checkpoint.pth.tar \
  --unfreeze_layer fc2 \
  --project resnet50-0 

# 2-3. resnet50_gradcam (no training)

# 2-4. resnet50_gradcam + ours - Check if `large_feature_map` is FALSE!
python main.py \
  --dataset_name ILSVRC \
  --architecture resnet50 \
  --wsol_method cam \
  --method gradcam \
  --experiment_name resnet50_gradcam_ours \
  --wandb_name resnet50_gradcam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/ \
  --large_feature_map FALSE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 3 \
  --workers 4 \
  --gpus 1 \
  --lr 0.005 \
  --weight_decay 1.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/resnet50/resnet50_gradcam/last_checkpoint.pth.tar \
  --unfreeze_layer fc2 \
  --project resnet50-0 

# 3-1. Inceptionv3_cam
python main.py \
  --dataset_name ILSVRC \
  --architecture inception_v3 \
  --wsol_method cam \
  --method cam \
  --experiment_name inceptionv3_cam \
  --wandb_name inceptionv3_cam \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/ \
  --large_feature_map TRUE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 3 \
  --workers 4 \
  --gpus 1 \
  --lr 0.0003606503 \
  --weight_decay 5.00E-04 \
  --model_structure vanilla \
  --project inceptionv3-0 

# 3-2. Inceptionv3_cam + ours
python main.py \
  --dataset_name ILSVRC \
  --architecture inception_v3 \
  --wsol_method cam \
  --method cam \
  --experiment_name inceptionv3_cam_ours \
  --wandb_name inceptionv3_cam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/ \
  --large_feature_map TRUE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 3 \
  --workers 4 \
  --gpus 1 \
  --lr 0.0005 \
  --weight_decay 5.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/inceptionv3/inceptionv3_cam/last_checkpoint.pth.tar \
  --unfreeze_layer SPG_A4_2 \
  --project inceptionv3-0 

# 3-3. Inceptionv3_gradcam (no training)

# 3-4. Inceptionv3_gradcam + ours - Check if `large_feature_map` is FALSE!
python main.py \
  --dataset_name ILSVRC \
  --architecture inception_v3 \
  --wsol_method cam \
  --method gradcam \
  --experiment_name inceptionv3_gradcam_ours \
  --wandb_name inceptionv3_gradcam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/ \
  --large_feature_map FALSE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 3 \
  --workers 4 \
  --gpus 1 \
  --lr 0.0005 \
  --weight_decay 5.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/imagenet/inceptionv3/inceptionv3_gradcam/last_checkpoint.pth.tar \
  --unfreeze_layer fc2 \
  --project inceptionv3-0 

# -------------------------------------------------------------- #
# cub
# 1-1. vgg16_cam
python main.py \
  --dataset_name CUB \
  --architecture vgg16 \
  --wsol_method cam \
  --method cam \
  --experiment_name vgg16_cam \
  --wandb_name vgg16_cam \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cub/ \
  --large_feature_map FALSE \
  --epoch 50 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.00001268269 \
  --weight_decay 5.00E-04 \
  --model_structure vanilla \
  --project vgg16-cub-0 

# 1-2. vgg16_cam + ours
python main.py \
  --dataset_name CUB \
  --architecture vgg16 \
  --wsol_method cam \
  --method cam \
  --experiment_name vgg16_cam_ours \
  --wandb_name vgg16_cam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cub/ \
  --large_feature_map FALSE \
  --epoch 50 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.00005 \
  --weight_decay 5.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/cub/vgg16/vgg16_cam/last_checkpoint.pth.tar \
  --unfreeze_layer fc2 \
  --project vgg16-cub-0 

# 1-3. vgg16_gradcam 
python main.py \
  --dataset_name CUB \
  --architecture vgg16 \
  --wsol_method cam \
  --method gradcam \
  --experiment_name vgg16_gradcam \
  --wandb_name vgg16_gradcam \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cub/ \
  --large_feature_map FALSE \
  --epoch 50 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.0001 \
  --weight_decay 5.00E-04 \
  --model_structure vanilla \
  --project vgg16-cub-0 

# 1-4. vgg16_gradcam + ours
python main.py \
  --dataset_name CUB \
  --architecture vgg16 \
  --wsol_method cam \
  --method gradcam \
  --experiment_name vgg16_gradcam_ours \
  --wandb_name vgg16_gradcam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cub/ \
  --large_feature_map FALSE \
  --epoch 50 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.001 \
  --weight_decay 5.00E-04 \
  --model_structure vanilla \
  --project vgg16-cub-0 


# 2-1. resnet50_cam
python main.py \
  --dataset_name CUB \
  --architecture resnet50 \
  --wsol_method cam \
  --method cam \
  --experiment_name resnet50_cam \
  --wandb_name resnet50_cam \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cub/ \
  --large_feature_map TRUE \
  --epoch 50 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.00023222617 \
  --weight_decay 1.00E-04 \
  --model_structure vanilla \
  --project resnet50-cub-0 

# 2-2. resnet50_cam + ours
python main.py \
  --dataset_name CUB \
  --architecture resnet50 \
  --wsol_method cam \
  --method cam \
  --experiment_name resnet50_cam_ours \
  --wandb_name resnet50_cam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cub/ \
  --large_feature_map TRUE \
  --epoch 50 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.0001 \
  --weight_decay 1.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/cub/resnet50/resnet50_cam/last_checkpoint.pth.tar \
  --unfreeze_layer fc2 \
  --project resnet50-cub-0 

# 2-3. resnet50_gradcam 
python main.py \
  --dataset_name CUB \
  --architecture resnet50 \
  --wsol_method cam \
  --method gradcam \
  --experiment_name resnet50_gradcam \
  --wandb_name resnet50_gradcam \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cub/ \
  --large_feature_map TRUE \
  --epoch 50 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.00023222617 \
  --weight_decay 1.00E-04 \
  --model_structure vanilla \
  --project resnet50-cub-0 

# 2-4. resnet50_gradcam + ours 
python main.py \
  --dataset_name CUB \
  --architecture resnet50 \
  --wsol_method cam \
  --method gradcam \
  --experiment_name resnet50_gradcam_ours \
  --wandb_name resnet50_gradcam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cub/ \
  --large_feature_map TRUE \
  --epoch 50 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.0001 \
  --weight_decay 1.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/cub/resnet50/resnet50_gradcam/last_checkpoint.pth.tar \
  --unfreeze_layer fc2 \
  --project resnet50-cub-0 

# ---------------------------------------------------------------- #

# cars
# 1-1. vgg16_cam
python main.py \
  --dataset_name CARS \
  --architecture vgg16 \
  --wsol_method cam \
  --method cam \
  --experiment_name vgg16_cam \
  --wandb_name vgg16_cam \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/ \
  --large_feature_map FALSE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.001 \
  --weight_decay 5.00E-04 \
  --model_structure vanilla \
  --project vgg16-cars-0 

# 1-2. vgg16_cam + ours
python main.py \
  --dataset_name CARS \
  --architecture vgg16 \
  --wsol_method cam \
  --method cam \
  --experiment_name vgg16_cam_ours \
  --wandb_name vgg16_cam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/ \
  --large_feature_map FALSE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.003 \
  --weight_decay 5.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/vgg16/vgg16_cam/last_checkpoint.pth.tar \
  --unfreeze_layer fc2 \
  --project vgg16-cars-0 

# 1-3. vgg16_gradcam
python main.py \
  --dataset_name CARS \
  --architecture vgg16 \
  --wsol_method cam \
  --method gradcam \
  --experiment_name vgg16_gradcam \
  --wandb_name vgg16_gradcam \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/ \
  --large_feature_map FALSE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.001 \
  --weight_decay 5.00E-04 \
  --model_structure vanilla \
  --project vgg16-cars-0 
  
# 1-4. vgg16_gradcam + ours
python main.py \
  --dataset_name CARS \
  --architecture vgg16 \
  --wsol_method cam \
  --method gradcam \
  --experiment_name vgg16_gradcam_ours \
  --wandb_name vgg16_gradcam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/ \
  --large_feature_map FALSE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.0001 \
  --weight_decay 5.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/vgg16/vgg16_gradcam/last_checkpoint.pth.tar \
  --unfreeze_layer classifier_2 \
  --project vgg16-cars-0 

# 2-1. resnet50_cam
python main.py \
  --dataset_name CARS \
  --architecture resnet50 \
  --wsol_method cam \
  --method cam \
  --experiment_name resnet50_cam \
  --wandb_name resnet50_cam \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/ \
  --large_feature_map TRUE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.001 \
  --weight_decay 1.00E-04 \
  --model_structure vanilla \
  --project resnet50-cars-0 

# 2-2. resnet50_cam + ours
python main.py \
  --dataset_name CARS \
  --architecture resnet50 \
  --wsol_method cam \
  --method cam \
  --experiment_name resnet50_cam_ours \
  --wandb_name resnet50_cam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/ \
  --large_feature_map TRUE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.0001 \
  --weight_decay 1.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/resnet50/resnet50_cam/last_checkpoint.pth.tar \
  --unfreeze_layer fc2 \
  --project resnet50-cars-0 

# 2-3. resnet50_gradcam
python main.py \
  --dataset_name CARS \
  --architecture resnet50 \
  --wsol_method cam \
  --method gradcam \
  --experiment_name resnet50_gradcam \
  --wandb_name resnet50_gradcam \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/ \
  --large_feature_map TRUE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.001 \
  --weight_decay 1.00E-04 \
  --model_structure vanilla \
  --project resnet50-cars-0 

# 2-4. resnet50_gradcam + ours
python main.py \
  --dataset_name CARS \
  --architecture resnet50 \
  --wsol_method cam \
  --method gradcam \
  --experiment_name resnet50_gradcam_ours \
  --wandb_name resnet50_gradcam_ours \
  --root /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/ \
  --large_feature_map TRUE \
  --epoch 10 \
  --batch_size 32 \
  --lr_decay_frequency 15 \
  --workers 4 \
  --gpus 1 \
  --lr 0.0001 \
  --weight_decay 1.00E-04 \
  --model_structure b2 \
  --ft_ckpt /home/yoojinoh/recam-imagenet-gradcam/experiment/cars/resnet50/resnet50_gradcam/last_checkpoint.pth.tar \
  --unfreeze_layer fc2 \
  --project resnet50-cars-0 
