"""
05_benchmark.py — compute image-similarity metrics over a folder of
                 reconstructed images, matched against their base images.

Thin CLI wrapper around
:func:`noise2params.noise_image.group_sim_metrics_2`.  Predicted files must
be named ``{base_image_stem}_{anything}.png`` so the base image can be
resolved by stem.

Example:

    python scripts/05_benchmark.py \\
        --save-folder ./reconstructions/test_saddle \\
        --name saddle_test_metrics --metrics all
"""

import argparse
from noise2params.noise_image import group_sim_metrics_2


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--save-folder', required=True,
                   help='Folder of predicted .png files.')
    p.add_argument('--name', default='metrics',
                   help='CSV filename stem.')
    p.add_argument('--metrics', default='all',
                   help="'classic' | 'modern' | 'deep learning' | 'all' | comma-separated list.")
    p.add_argument('--recompute', action='store_true',
                   help='Force recomputation of all requested metrics.')
    args = p.parse_args()

    metrics_arg = args.metrics
    if ',' in metrics_arg:
        metrics_arg = [m.strip() for m in metrics_arg.split(',') if m.strip()]
    group_sim_metrics_2(save_folder=args.save_folder, name_str=args.name,
                        metrics=metrics_arg, recompute=args.recompute)


if __name__ == '__main__':
    main()
