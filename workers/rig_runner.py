"""Subprocess entry for the UniRig stages (run by workers/rig_worker.py with envs/rig/bin/python).

  rig_runner.py <workdir> <task.yaml> <seed>

`workdir` (cwd) holds a copy of UniRig's configs/ and dataset_inference_clean/job/<npz>; vendor/UniRig stays untouched and only
provides the Python package `src` and the run.py driver. The CUDA / Blender stand-ins of workers/rig_shims.py are installed
first. Everything the stage prints goes to stdout (the worker parses `PROGRESS` lines).
"""
import os
import runpy
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
STUDIO = os.environ.get("STUDIO_ROOT") or os.path.dirname(HERE)
UNIRIG = os.path.join(STUDIO, "vendor", "UniRig")


def main():
    workdir, task, seed = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, HERE)
    sys.path.insert(0, UNIRIG)
    os.chdir(workdir)
    import rig_shims
    rig_shims.install()
    rig_shims.patch_voxelization()
    sys.argv = ["run.py", f"--task={task}", f"--seed={seed}", "--npz_dir=tmp"]
    runpy.run_path(os.path.join(UNIRIG, "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
