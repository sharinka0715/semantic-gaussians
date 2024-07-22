# Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting

<p align="left">
    <a href='https://arxiv.org/pdf/2403.15624.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/abs/2403.15624'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://semantic-gaussians.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

This repository is the official implemetation of the paper "[Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting](https://arxiv.org/abs/2403.15624)".

<div align=center>
<img src='./assets/teaser.png' width=80%>
</div>

## Abstract

Open-vocabulary 3D scene understanding presents a significant challenge in computer vision, with wide-ranging applications in embodied agents and augmented reality systems. Previous approaches have adopted Neural Radiance Fields (NeRFs) to analyze 3D scenes. In this paper, we introduce Semantic Gaussians, a novel open-vocabulary scene understanding approach based on 3D Gaussian Splatting. Our key idea is distilling pretrained 2D semantics into 3D Gaussians. We design a versatile projection approach that maps various 2D semantic features from pretrained image encoders into a novel semantic component of 3D Gaussians, without the additional training required by NeRFs. We further build a 3D semantic network that directly predicts the semantic component from raw 3D Gaussians for fast inference. We explore several applications of Semantic Gaussians: semantic segmentation on ScanNet-20, where our approach attains a 9.3\% mIoU and 6.5\% mAcc improvement over prior open-vocabulary scene understanding counterparts; object part segmentation, scene editing, and spatial-temporal segmentation with better qualitative results over 2D and 3D baselines, highlighting its versatility and effectiveness on supporting diverse downstream tasks.

## Prerequisites

This code has been tested on Ubuntu 22.04 and NVIDIA RTX 4090. We recommend to use Linux and an NVIDIA GPU with ≥ 16GB VRAM. This repository may support Windows machines but it was not evaluated. It cannot support MacOS system as it requires CUDA.

## Install

1. Clone our repository (remember to add the `--recursive` argument to clone submodules).

    ```bash
    git clone https://github.com/sharinka0715/semantic-gaussians --recursive
    cd semantic-gaussians
    ```

2. Create individual virtual environment (or use existing environments with CUDA Development kit and corresponding version of PyTorch).
    ```bash
    conda env create -f environment.yaml
    conda activate sega
    ```

3. Install additional dependencies with pip as many of them need to be compiled.
    ```bash
    pip install -r requirements.txt
    ```

4. Compile and install MinkowskiEngine through anaconda, recommending to install through [official instructions](https://github.com/NVIDIA/MinkowskiEngine?tab=readme-ov-file#installation).
    ```bash
    # Here is an example only for Anaconda, CUDA 11.x
    conda install openblas-devel -c anaconda
    pip install git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
    ```

## Prepare Dataset and Pretrained 2D Models

### Data structure

This repository supports three formats of dataset for 3D Gaussians Splatting:
* Blender format
    ```
    scene_name
    |-- images/
    |-- points3d.ply
    |-- transforms_train.json
    ```
* COLMAP format
    ```
    scene_name
    |-- images/
    |-- sparse/
    |   |-- 0
    |   |   |-- cameras.bin
    |   |   |-- points3D.bin
    ```
* ScanNet format
    ```
    scene_name
    |-- color/
    |-- intrnsic/
    |-- pose/
    |-- points3d.ply
    ```
Blender and COLMAP formats are originally supported by 3D Gaussian Splatting and many NeRF-based works. You can easily prepare your dataset as these two format.

The ScanNet dataset can be extracted by `tools/scannet_sens_reader.py`. You can also use `tools/unzip_lable_filt.py` to extract ground truth semantic labels in ScanNet-20 dataset.
```bash
# An example used for experiments in paper
python tools/scannet_sens/reader.py --input_path /PATH/TO/YOUR/scene0000_00 --output_path /PATH/TO/YOUR/OUTPUT/scene0000_00
```

### Datasets Used in Paper
| Dataset Name | Download Link | Format |
|----|----|----|
| ScanNet | [Official GitHub link](https://github.com/ScanNet/ScanNet) | ScanNet (need pre-process) |
| MVImgNet | [Official GitHub link](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet) | COLMAP |
| CMU Panoptic | [Official Page](http://domedb.perception.cs.cmu.edu/), [Dynamic 3D Gaussians Page](https://dynamic3dgaussians.github.io/) | Other (need pre-process) |
| Mip-NeRF 360 | [Official Project Page](https://jonbarron.info/mipnerf360/) | COLMAP |

### Pretrained 2D Vision-Language Models

You should put these downloaded pretrained checkpoints under the `./weight/` folder, or you can modify the saving path in YAML configs.

| Model Name | Checkpoint | Download Link |
|----|----|----|
| CLIP | ViT-L/14@336px | Automatically download by `openai/CLIP` |
| OpenSeg | Default | [Google Drive](https://drive.google.com/file/d/1DgyH-1124Mo8p6IUJ-ikAiwVZDDfteak/view), [Official Repo](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/openseg)
| LSeg | Model for Demo | [Google Drive](https://drive.google.com/file/d/1FTuHY1xPUkM-5gaDtMfgCl3D0gR89WV7/view?usp=sharing), [Official Repo](https://github.com/isl-org/lang-seg)
| SAM | ViT-H | [Direct Link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), [Official Repo](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
| VLPart | Swin-Base | [Direct Link](https://github.com/Cheems-Seminar/grounded-segment-any-parts/releases/download/v1.0/swinbase_part_0a0000.pth), [Grounded Segment Any Parts Repo](https://github.com/Cheems-Seminar/grounded-segment-any-parts?tab=readme-ov-file#model-checkpoints)

## Usage

This repository has 4 entries to start a program. Every entry has its corresponding config YAML file. You only need to run `python xxx.py`, all configs are in YAML files.

* `train.py`: Train normal RGB gaussians. Code mainly from 3D Gaussian Splatting official repository.
    
    config: `config/official_train.yaml`.
    
    This will output 3D Gaussians under `output/` folder.

* `fusion.py`: Apply 2D versatile projection.
    
    config: `config/fusion_scannet.yaml`.

    This will output fused features under `config.fusion.out_dir`

* `distill.py`: Train 3D semantic network.
    
    config: `config/distill_scannet.yaml`.

    This will output 3D semantic network checkpoints in `results_distlll/` folder.

* `eval_segmentation.py`: Evaluate the semantic segmentation performance on ScanNet dataset.
    
    config: `config/eval.yaml`.

    This will print the evaluation results on the screen.

* `view_viser.py`: View the semantic Gaussians. Need 2D projected results (*.pt) and original RGB Gaussians.
    
    config: `config/view_scannet.yaml`.

    This will open a web service supported by [viser](https://github.com/nerfstudio-project/viser).

## Acknowledgements

We appreciate the works below as this repository is heavily based on them:

[[SIGGRAPH 2023] 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)

[[CVPR 2023] OpenScene: 3D Scene Understanding with Open Vocabularies](https://github.com/pengsongyou/openscene)

[[ECCV 2022] OpenSeg: Scaling Open-Vocabulary Image Segmentation with Image-Level Labels](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/openseg/)

[[Cheems Seminar] Grounded Segment Anything: From Objects to Parts](https://github.com/Cheems-Seminar/grounded-segment-any-parts)


## News

- [2024.07] We fix some dependency problems in our code. Add LSeg modules.

- [2024.05] We release our initial version of implemetation.

## Citation

```bibtex
@misc{guo2024semantic,
    title={Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting}, 
    author={Jun Guo and Xiaojian Ma and Yue Fan and Huaping Liu and Qing Li},
    year={2024},
    eprint={2403.15624},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
```
