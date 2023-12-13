
This code is adapted from GraphGPS:https://github.com/rampasek/GraphGPS
### Python environment setup (pyg_lib only works for Linux, can also potentially run on windows without pyg_lib)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install fsspec rdkit

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

```


### Running
```bash

python main.py --cfg configs/GPS/neural-<ALTERNATIVE_DATASET>-<GPS/Trans>+RWSE.yaml  wandb.use False

```

replace <ALTERNATIVE_DATASET> with either Act/FI/WM
replace <GPS/Trans> with GPS/Trans for using full GPS model or only transformers

## Unit tests (test if environment works)

To run all unit tests, execute from the project root directory:

```bash
python -m unittest -v
```


```
