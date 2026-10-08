# Generator

This folder writes `../phases/`. Students do not work here.

`before/` is the unfinished package. `layers.toml` says which file each
workshop owns. `build_layers.py` turns those two into one `before/` and
`after/` per workshop. `verify_layers.py` checks the result.

```bash
python3 build_layers.py
python3 verify_layers.py
```
