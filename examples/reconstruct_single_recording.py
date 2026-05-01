"""
examples/reconstruct_single_recording.py

Run a trained reconstruction CNN on a single Prophesee `.parquet`
recording.  Displays the reconstructed image.

Usage:
    python examples/reconstruct_single_recording.py path/to/rec.parquet path/to/model.ckpt
"""

import sys
from noise2params.noise_image import infer_recording


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    rec_path = sys.argv[1]
    ckpt_path = sys.argv[2]

    infer_recording(
        rec_file=rec_path,
        CKPT=ckpt_path,
        integration_time_s=5,
        rand_seg=(5, 4),      # 5 s random segment, starting at least 4 s in
        remove_outliers=True,
        plot=True,
    )


if __name__ == '__main__':
    main()
