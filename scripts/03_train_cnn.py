"""
03_train_cnn.py — launch the reconstruction-CNN training script.

This is a thin wrapper that dispatches to one of two launchers depending on
``--target``:

    * ``native`` → :func:`noise2params.utils.run_train_synthetic_6`
        Runs :file:`noise2image/train_synthetic_6.py` under a local Python
        interpreter.  Usable on Linux / macOS / Windows.

    * ``wsl`` → :func:`noise2params.utils.run_train_synthetic_6_wsl`
        Runs the same script inside WSL (Windows Subsystem for Linux)
        using a Linux virtualenv.  **This is the path that produced the
        Noise2Params paper results.**  Linux/WSL is recommended.

Either way, the underlying command is
``python train_synthetic_6.py --data_folder <X> --val_data_folders <Y> ...``.

See ``configs/training_published.yaml`` for the exact flag values used
for the published checkpoints.

IMPORTANT
---------
* Use ``--batch-size 2``.  See the caveat in the repository README: larger
  batch sizes have been observed to cause multi-order-of-magnitude
  training slowdowns on both our development GPUs.  We believe this is
  a latent bug in the U-Net + attention architecture; we did not
  resolve it.  Reproducing the paper requires ``--batch-size 2``.

* Installing the training environment is finicky.  If you encounter
  errors in flash-attention, denoising-diffusion-pytorch, or torchmetrics
  at the first forward pass, try (inside the training venv)::

      pip uninstall -y denoising-diffusion-pytorch
      pip install denoising-diffusion-pytorch==1.10.7
      pip install "torchmetrics<1.8.0"

  Upgrading/downgrading unrelated packages has occasionally re-broken this
  in our setup and we had to re-apply the fix.  See the repository README.
"""

import argparse
from noise2params.utils import run_train_synthetic_6, run_train_synthetic_6_wsl


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', choices=['native', 'wsl'], default='wsl',
                   help='Launcher: native Python or WSL.  Default wsl (paper path).')
    p.add_argument('--data-folder', required=True,
                   help='Training dataset folder (e.g. data_5e6_5).')
    p.add_argument('--val-data-folders', nargs='+', required=True,
                   help='Validation dataset folder(s).')
    p.add_argument('--num-epochs', default='40')
    p.add_argument('--batch-size', default='2',
                   help='Leave at 2 to reproduce paper; see docstring caveat.')
    p.add_argument('--lr', default='5e-5')
    p.add_argument('--integration-time', default='5')
    p.add_argument('--resume-from', default=None)
    # native-only
    p.add_argument('--python-executable', default=None,
                   help='(native only) Path to the training venv python.')
    # wsl-only
    p.add_argument('--wsl-project-dir', default='~/projects/noise2image',
                   help='(wsl only) WSL-side path to noise2image/.')
    p.add_argument('--wsl-venv-python', default='~/.venvs/n2i/bin/python',
                   help='(wsl only) WSL-side path to python in the training venv.')

    args = p.parse_args()

    if args.target == 'native':
        run_train_synthetic_6(
            data_folder=args.data_folder,
            python_executable=args.python_executable,
            val_data_folders=args.val_data_folders,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            extra_args=['--lr', args.lr,
                        '--integration_time', args.integration_time] +
                       (['--resume_from', args.resume_from] if args.resume_from else []),
        )
    else:
        run_train_synthetic_6_wsl(
            data_folders=args.data_folder,
            val_data_folders=args.val_data_folders,
            num_epochs=args.num_epochs,
            resume_from=args.resume_from,
            lr=args.lr,
            wsl_project_dir=args.wsl_project_dir,
            wsl_venv_python=args.wsl_venv_python,
            integration_time=args.integration_time,
        )


if __name__ == '__main__':
    main()
