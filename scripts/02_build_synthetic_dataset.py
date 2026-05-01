"""
02_build_synthetic_dataset.py — build a (base-image, synthetic-noise-image)
                               pair database under a chosen probability model.

Thin CLI wrapper around
:func:`noise2params.noise_image.synthetic_data_base_maker`.

Published datasets:

    ``data_5e6_5``  → --model saddle
    ``data_5e6_6``  → --model Gaussian
    (no shipped dataset for Poisson — see README; code path retained
     for reproducibility.)

Example (saddle-point, reproducing ``data_5e6_5``):

    python scripts/02_build_synthetic_dataset.py \\
        --source "Data/Noise2Params Dataset/indist_images_full" \\
        --event-target "Data/Noise2Params Dataset/data_5e6_5/synth_events" \\
        --image-target "Data/Noise2Params Dataset/data_5e6_5/synth_images" \\
        --model saddle

All fit coefficients default to the published values (see
:file:`configs/params_published.yaml`).
"""

import argparse
from noise2params.noise_image import synthetic_data_base_maker


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True,
                   help='Folder of base greyscale .png/.jpg images.')
    p.add_argument('--event-target', required=True,
                   help='Output folder for events_all.npy.')
    p.add_argument('--image-target', required=True,
                   help='Output folder for images_all.npy.')
    p.add_argument('--model', choices=['saddle', 'Gaussian', 'Poisson'], default='Poisson',
                   help='Probability model.  Published: saddle (main) / Gaussian (ablation).')
    p.add_argument('--num-variations', type=int, default=2,
                   help='Random augmentations per source image.  Published: 2.')
    p.add_argument('--save-format', choices=['memmap', 'hdf5', 'npy'], default='memmap',
                   help='Output format.  memmap matches precomputed_dataset loader.')
    p.add_argument('--crop-or-resize', choices=['rand', 'crop', 'resize'], default='rand',
                   help='How to reduce source images >1280x720.')
    p.add_argument('--brightness-aug', action='store_true',
                   help='Apply in-[0.7, 1.3] brightness multiplier.  Published: off.')
    p.add_argument('--no-plot', action='store_true',
                   help='Suppress the preview plots of the first few samples.')
    args = p.parse_args()

    synthetic_data_base_maker(
        source_folder=args.source,
        event_target_folder=args.event_target,
        image_target_folder=args.image_target,
        num_variations=args.num_variations,
        noise_modeler_kwargs={'model': args.model},
        save_format=args.save_format,
        apply_brightness_aug=args.brightness_aug,
        crop_or_resize=args.crop_or_resize,
        Plot=not args.no_plot,
    )


if __name__ == '__main__':
    main()
