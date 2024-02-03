#!/bin/bash
# t=0
# gpu=$1
# for scene in $(ls /scratch/ml/guojun/scannet_blender/train/)
# do
#     t=$[$t+1]
#     if [ $[$t%8] -eq $gpu ];
#     then
#         echo $t, ${scene}
#         srun -N 1 --cpus-per-task 8 -t 24:00:00 -p DGX --qos=lv0b --gres=gpu:1 --account=research --mem=32GB \
#             python fusion.py \
#             scene.scene_path="/scratch/ml/guojun/scannet_blender/train/${scene}" \
#             model.model_dir="/scratch/ml/guojun/scannet_gaussians/train/${scene}" \
#             fusion.out_dir="/scratch/ml/guojun/scannet_fusion/train/${scene}"
#     fi
# done

# conda activate gasp;cd /scratch/guojun/semantic_gaussians

# export CXX=c++; python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
# salloc -N 1 --cpus-per-task 8 -t 20:00:00 -p DGX --qos=lv0b --gres=gpu:1 --account=research --mem=32GB

# python fusion.py \
#     scene.scene_path="/scratch/ml/guojun/scannet_blender/train/scene0000_00" \
#     model.model_dir="/scratch/ml/guojun/scannet_gaussians/train/scene0000_00" \
#     fusion.out_dir="/scratch/ml/guojun/scannet_fusion/train/scene0000_00"

# conda activate colmapenv;cd /scratch/guojun/semantic_gaussians

# srun -N 1 --cpus-per-task 8 -t 24:00:00 -p DGX --qos=lv0b --gres=gpu:1 --account=research --mem=32GB python distill.py 

# t=0
# gpu=$1
# for scene in $(ls /scratch/masaccio/existing_datasets/scannet/scans/)
# do
#     t=$[$t+1]
#     if [ $[$t%8] -eq $gpu ];
#     then
#         echo $t, ${scene}
#         srun -N 1 --cpus-per-task 8 -t 24:00:00 -p DGX --qos=lv0b --account=research --mem=32GB \
#             python tools/scannet_sens_reader.py \
#             --input_path /scratch/masaccio/existing_datasets/scannet/scans/${scene} \
#             --output_path /scratch2/ml/guojun/scannet_half/${scene} \
#             --export_width 648 \
#             --export_height 484 \
#             --not_export_depth_images
#     fi
# done

# t=0
# gpu=$1
# for scene in $(ls /scratch2/ml/guojun/scannet_dataset/)
# do
#     t=$[$t+1]
#     if [ $[$t%8] -eq $gpu ];
#     then
#         echo $t, ${scene}
#         srun -N 1 --cpus-per-task 8 -t 24:00:00 -p DGX --qos=lv0b --account=research --gres=gpu:1 --mem=32GB \
#             python tools/run_colmap.py \
#             --input_path /scratch2/ml/guojun/scannet_dataset/${scene}
#     fi
# done

t=0
gpu=$1
for scene in $(ls /scratch2/ml/guojun/scannet_half/val/)
do
    t=$[$t+1]
    if [ $[$t%8] -eq $gpu ];
    then
        echo $t, ${scene}
        srun -N 1 --cpus-per-task 8 -t 24:00:00 -p DGX --qos=lv0b --account=research --gres=gpu:1 --mem=32GB \
            python train.py scene.scene_path="/scratch2/ml/guojun/scannet_half/val/${scene}" \
            train.exp_name=scannet_val/${scene}
    fi
done

# t=0
# gpu=$1
# for scene in $(ls /scratch2/ml/guojun/scannet_half/)
# do
#     t=$[$t+1]
#     if [ $[$t%8] -eq $gpu ];
#     then
#         echo $t, ${scene}
#         srun -N 1 --cpus-per-task 8 -t 24:00:00 -p DGX --qos=lv0b --account=research --gres=gpu:1 --mem=32GB \
#             python fusion.py scene.scene_path="/scratch2/ml/guojun/scannet_half/${scene}" \
#             model.model_dir=./output/scannet/${scene} \
#             fusion.out_dir=./output/scannet_fusion/${scene}
#         break
#     fi
# done