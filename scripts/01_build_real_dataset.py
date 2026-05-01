"""
01_build_real_dataset.py — build a (base-image, real-event-count-image)
                          pair database from event-camera recordings.

Thin CLI wrapper around :func:`noise2params.noise_image.real_data_base_maker`.

Typical invocation for reproducing the shipped ``data_real_5e6_2``
(training set, in-distribution images, real recordings) and
``validation_5e6_5`` (OOD DIV2K validation images, real recordings):

    python scripts/01_build_real_dataset.py \\
        --mapping-csv "19-38-49_to_00-17-48_recording_data.csv" \\
        --image-folder "Data/Noise2Params Dataset/indist_images_full" \\
        --event-target "Data/Noise2Params Dataset/data_real_5e6_2/synth_events" \\
        --image-target "Data/Noise2Params Dataset/data_real_5e6_2/synth_images" \\
        --integration-time 5 --rand-seg 5 4

The ``--mapping-csv`` is one of the three mapping CSVs shipped with the
public dataset; see :file:`docs/DATA.md` for which CSV goes with which
image set.  All path arguments may be bare stems: they are resolved
project-wide by :func:`noise2params.utils.file_checker` where possible.
"""

import argparse
from noise2params.noise_image import real_data_base_maker


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mapping-csv', required=True,
                   help='Recording / base-image mapping CSV (see docs/DATA.md).')
    p.add_argument('--image-folder', required=True,
                   help='Folder of base greyscale .png images.')
    p.add_argument('--event-target', required=True,
                   help='Output folder for events_all.npy.')
    p.add_argument('--image-target', required=True,
                   help='Output folder for images_all.npy.')
    p.add_argument('--integration-time', type=int, default=5,
                   help='Event integration window in seconds (paper: 5).')
    p.add_argument('--rand-seg', nargs=2, type=int, default=[5, 4],
                   metavar=('LEN_S', 'START_S'),
                   help='Random-segment extraction: (length_s, start_offset_s).')
    p.add_argument('--num-variations', type=int, default=1,
                   help='Random segments per recording.')
    p.add_argument('--use-full-time', action='store_true',
                   help='Instead of random segments, tile the full recording.')
    args = p.parse_args()

    real_data_base_maker(
        image_rec_mapping=args.mapping_csv,
        val_pair_folder=args.image_folder,
        event_target_folder=args.event_target,
        image_target_folder=args.image_target,
        integration_time_s=args.integration_time,
        use_full_time=args.use_full_time,
        rand_seg=None if args.use_full_time else tuple(args.rand_seg),
        num_variations=args.num_variations,
    )


if __name__ == '__main__':
    main()
