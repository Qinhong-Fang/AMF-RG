# AMF-RG: Cross-Source Point Cloud Registration

This repository contains the PyTorch implementation of **AMF-RG (Adaptive Multi-scale Fusion and Matching Residual Guidance)**, built upon [GeoTransformer](https://github.com/qinzheng93/GeoTransformer). It enhances the fine-matching stage with adaptive feature fusion (AF) and residual-guided score refinement (RG).

## Environment

The experiments were conducted on Linux with an NVIDIA GPU using the following **GeoTransformer** environment:

| Component | Version |
| --- | --- |
| Conda environment | `geotransformer` |
| Python | `3.9.0` |
| PyTorch | `2.9.0.dev20250709+cu128` (nightly build) |
| CUDA (PyTorch build) | `12.8` |

Activate the existing environment and verify the versions:

```bash
conda activate geotransformer
python -c "import sys, torch; print('Python:', sys.version); print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.is_available())"
```

To create a fresh Python 3.9 environment:

```bash
conda create -n geotransformer python=3.9 -y
conda activate geotransformer

pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

pip install -r requirements.txt
python setup.py build develop
```

## Dataset

Prepare the KITTI Cross-Source point clouds and metadata in `data/Kitti/` using the same dataset split as the experiments. The dataset is not included in this repository.

## Training

```bash
cd experiments/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

Both modules are controlled in `config.py`:

```python
_C.fine_matching.use_adaptive_fusion = True
_C.fine_matching.use_residual_refinement = True
```

## Evaluation

From the same experiment directory, evaluate a checkpoint saved for an epoch:

```bash
CUDA_VISIBLE_DEVICES=0 ./eval.sh EPOCH
```

Replace `EPOCH` with the checkpoint epoch number. To evaluate a compatible checkpoint by path:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../../weights/kitti.pth.tar
CUDA_VISIBLE_DEVICES=0 python eval.py --method=lgr
```

## Acknowledgements

Our implementation is based on [GeoTransformer](https://github.com/qinzheng93/GeoTransformer). We thank the original authors for releasing their code.
