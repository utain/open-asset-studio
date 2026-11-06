# Contributing

Thanks for helping. Issues and pull requests are welcome.

- **Run the tests:** `envs/core/bin/python -m pytest -q`. They need only `envs/core`, with no GPU and no weights.
- **Add a model:** see "Add a model" in the README and `docs/CONTRACTS.md` §1–2. Put new entries in a
  `models.d/<group>.yaml`, give the model its own backend venv when its dependencies conflict, add
  `env-lock/<backend>.txt` (`uv pip freeze`), pin any upstream repo in `vendor/vendor.lock`, and record the
  measured memory, load and run time and the exact licence (with `commercial: false` for non-commercial weights).
- **Hardware notes:** most measurements come from one DGX Spark (aarch64). Results from other GPUs are useful:
  please include the GPU, driver, and memory and time numbers.
- **Do not commit** model weights, generated audio or images, `data/`, `envs/` or `vendor/` checkouts.

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
