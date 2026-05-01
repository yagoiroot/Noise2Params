"""
04_infer.py — run a trained reconstruction CNN on event recordings.

Two modes, selected by presence of ``--mapping-csv``:

    * Single-recording mode:
          python scripts/04_infer.py --ckpt X.ckpt --recording path/to/rec.parquet

    * Group mode (drive off a recording/base-image mapping CSV):
          python scripts/04_infer.py --ckpt X.ckpt \\
              --mapping-csv "12-08-40_to_12-36-01_recording_data.csv" \\
              --save-folder ./reconstructions/test \\
              --integration-time 5 --rand-seg 5 4
"""

import argparse
from noise2params.noise_image import infer_recording, group_infer_recording


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True,
                   help='Path to the Lightning .ckpt.')
    p.add_argument('--recording', default=None,
                   help='Single recording to reconstruct (parquet).')
    p.add_argument('--mapping-csv', default=None,
                   help='Recording / base-image mapping CSV for group mode.')
    p.add_argument('--save-folder', default='./reconstructions',
                   help='Output folder for group-mode reconstructions.')
    p.add_argument('--integration-time', type=int, default=5)
    p.add_argument('--rand-seg', nargs=2, type=int, default=[5, 4],
                   metavar=('LEN_S', 'START_S'),
                   help='Random segment extraction: (length_s, start_offset_s).')
    p.add_argument('--name-suffix', default='',
                   help='Appended to output filenames in group mode.')
    p.add_argument('--no-plot', action='store_true',
                   help='Do not display the reconstruction (single-file mode).')
    args = p.parse_args()

    if args.mapping_csv is not None:
        group_infer_recording(
            file_list_csv=args.mapping_csv,
            save_folder=args.save_folder,
            checkpoint_path=args.ckpt,
            name_str=args.name_suffix,
            integration_time_s=args.integration_time,
            rand_seg=tuple(args.rand_seg),
        )
    elif args.recording is not None:
        infer_recording(
            rec_file=args.recording,
            CKPT=args.ckpt,
            integration_time_s=args.integration_time,
            rand_seg=tuple(args.rand_seg),
            plot=not args.no_plot,
        )
    else:
        p.error('must supply either --recording or --mapping-csv')


if __name__ == '__main__':
    main()
