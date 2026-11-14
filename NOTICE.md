# Notices

## License scope

The original source code of the authors (Python files under `pipeline/` and `src/`; configuration and environment
specification files under `configs/` and `environment/`) is released under the MIT License (`LICENSE`).

The aggregate results, figures, verification evidence and documentation (`data/`, `outputs/`, `evidence/`, Markdown files)
are not covered by the MIT License; all rights are reserved by the authors. They may be used to inspect, reproduce and
cite the accompanying paper.

## Third-party material

| Material | Source | Terms |
|---|---|---|
| LAMDA dataset (Baseline configuration), revision `ad9614bdd5556767f97ced2fce797c2f06408ebf`, IQSeC Lab | <https://huggingface.co/datasets/IQSeC-Lab/LAMDA/tree/ad9614bdd5556767f97ced2fce797c2f06408ebf> | Distributed by its owners under the license stated on the dataset card (MIT). Not redistributed here; only aggregate statistics computed from it are published. The MIT License of this repository does not license the dataset. |
| LAMDA code, commit `7728bfafcd5539a286b5f8c47b6f1e3b2d1f4249`, IQSeC Lab | <https://github.com/IQSeC-Lab/LAMDA/tree/7728bfafcd5539a286b5f8c47b6f1e3b2d1f4249> | Not included; its repository does not state a license at that commit. |
| Python packages (NumPy, pyarrow, scikit-learn, SciPy, LightGBM, XGBoost, PyTorch, psutil, matplotlib) | PyPI, PyTorch index | Installed by the user under their own licenses; not vendored. |
