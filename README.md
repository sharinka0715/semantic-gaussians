# Open-vocabulary scene understanding and editing with 3D Gaussian Splatting

## Install

```bash
# 1. Create individual virtual environment (or use existing environments with CUDA Development kit and corresponding version of PyTorch)
conda create -f environment.yaml
conda activate sega
# 2. Install additional dependencies with pip (many of them need to be compiled)
pip install -r requirements.txt
# 3. Compile and install MinkowskiEngine through anaconda, recommending to install through official instructions
conda install openblas-devel -c anaconda
pip install git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
```

## Usage
* `train.py` and `train_batch.py`: Train normal RGB gaussians. Code mainly from 3D Gaussian Splatting official repository.

* `fusion.py`: Apply 2D-3D projection.

* `distill.py`: Train 3D MinkowskiNet.

* `evaluate.py`: Evaluate the semantic segmentation performance on ScanNet dataset.

* `view_viser.py`: A visualization demo.

* Files in `tools/`: Some preprocessing code for some datasets.