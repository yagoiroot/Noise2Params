"""
examples/generate_one_synthetic_image.py

Minimal demo: generate a single synthetic noise-image from a base image
using the published parameter values, under the saddle-point model, at 5 s
integration time.

Usage:
    python examples/generate_one_synthetic_image.py path/to/any.png

Does not save anything by default; displays the synthetic noise image via
matplotlib.
"""

import sys
from noise2params.noise_image import event_noise_image_modeler


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    image_path = sys.argv[1]

    # Overrides match the published parameter values in
    # configs/params_published.yaml.
    event_noise_image_modeler(
        input_image_location=image_path,
        noise_modeler_kwargs={
            'time_steps': 5 * 1e6,
            'model': 'saddle',
            'B': 0.15,
            'B_sigma': 0.00651,
            'dead_time': 79,
            'alpha': 4.5,
            'uniform_pixels': False,
            'approach': 'NB sampling',
            'cap_or_norm': 'norm',
            'combine_polarities': True,
            'plot': False,
        },
        Plot=True,
    )


if __name__ == '__main__':
    main()
