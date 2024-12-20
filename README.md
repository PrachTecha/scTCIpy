# scTCI (single-cell Transition-cells-identification) Python

This repository hold the package that is a reimplementation of [CellTran](https://github.com/KChen-lab/transition-cells-identification.git) in Python with fast run time and scalability for larg number of cells. The package works seamlessly with AnnData object and Scanpy. The purpose of the package is to identify transition cell using stochastic differential equations (SDEs).

If you find this package useful to your research, please consider citing the original paper.
```
Wang, Y., Dede, M., Mohanty, V., Dou, J., et al. (2024).
A statistical approach for systematic identification of transition cells from scRNA-seq data.
Cell Reports Methods, 12(100913). https://doi.org/10.1016/j.crmeth.2024.100913 
```

Please use conda or mamba to create the environment.

```sh
mamba create -n scTCI python=3.11 scanpy numpy leidenalg cython ipykernel -c conda-forge -y
mamba activate scTCI
```

After creating and activating the environment, the scTCI package can be install by running the following command.

```sh
pip install git+https://github.com/skggm/skggm.git --no-deps
pip install git+https://github.com/PrachTecha/scTCIpy.git
```

The package can be loaded and run in Python with below example. It is recommended to nomalize the data and find clusters prior to running `.tl.transition_index`.

```python
import scTCIpy as stpy
stpy.tl.transition_index(adata)
```

Please refer to [test_nb.ipynb](./test_nb.ipynb) for example use.