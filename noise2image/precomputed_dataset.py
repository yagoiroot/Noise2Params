"""
noise2image.precomputed_dataset — load precomputed (events, image)
pairs for training.

Used by :file:`train_synthetic_6.py` to load datasets produced by
:func:`noise2params.noise_image.synthetic_data_base_maker` (synthetic)
and :func:`noise2params.noise_image.real_data_base_maker` (real).

Expected layout per dataset folder::

    <root>/
        synth_events/
            events_all.npy     shape (N, 2, H, W)  float32
            sample_names.txt
        synth_images/
            images_all.npy     shape (N, H, W)     float32 in [0, 1]

Three storage formats are supported:
    * ``memmap`` (default): raw ``.npy`` files, loaded as
      :class:`numpy.memmap`.  Fastest on local SSD.
    * ``hdf5``: a single gzip-compressed HDF5 file.  Smaller on disk.
    * ``npy``: one file per sample.  Most flexible but slowest.

Also exports :class:`BinomialSampling` — the on-the-fly transform used
when ``--on_fly_sampling`` is passed to the training script (resample
event counts from stored probability arrays each epoch).  The
Noise2Params paper uses the precomputed-counts branch; the on-the-fly
branch is available as an alternative training mode.
"""
# """Dataset loaders for precomputed synthetic events and images - supports multiple formats."""
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from pathlib import Path
# from PIL import Image
#
#
# class PrecomputedSyntheticDataset(Dataset):
#     """
#     Dataset that loads precomputed synthetic events and corresponding images.
#     Supports individual .npy files (original format).
#     """
#
#     def __init__(self, events_folder, images_folder, transform=None):
#         """
#         Args:
#             events_folder: Folder containing .npy event files (shape: 2, H, W)
#             images_folder: Folder containing .png image files
#             transform: Optional transform (typically just EventCountNormalization)
#         """
#         self.events_folder = Path(events_folder)
#         self.images_folder = Path(images_folder)
#         self.transform = transform
#
#         # Get all event files (.npy)
#         event_files = sorted(self.events_folder.glob('*.npy'))
#
#         # Filter out the memory-mapped files if they exist
#         event_files = [f for f in event_files if f.stem not in ['events_all', 'images_all']]
#
#         # Verify matching image files exist
#         self.sample_names = []
#         for event_file in event_files:
#             stem = event_file.stem
#             image_file = self.images_folder / f"{stem}.png"
#
#             if image_file.exists():
#                 self.sample_names.append(stem)
#             else:
#                 print(f"Warning: No matching image for {event_file.name}")
#
#         print(f"Loaded {len(self.sample_names)} precomputed samples (.npy format)")
#         print(f"  Events: {self.events_folder}")
#         print(f"  Images: {self.images_folder}")
#
#     def __len__(self):
#         return len(self.sample_names)
#
#     def __getitem__(self, idx):
#         """
#         Load precomputed event and image pair.
#
#         Returns:
#             events: (C, H, W) tensor of synthetic event counts
#             target: (1, H, W) tensor of target image
#             integration_time: scalar tensor (always 1.0)
#         """
#         sample_name = self.sample_names[idx]
#
#         # Load events from .npy (single array file)
#         event_path = self.events_folder / f"{sample_name}.npy"
#         events = np.load(event_path)  # Directly returns array, shape: (2, H, W)
#
#         # Ensure float32 and contiguous
#         if events.dtype != np.float32:
#             events = events.astype(np.float32)
#         if not events.flags['C_CONTIGUOUS']:
#             events = np.ascontiguousarray(events)
#
#         # Load image from .png
#         image_path = self.images_folder / f"{sample_name}.png"
#         image = np.array(Image.open(image_path).convert('L'))  # Grayscale
#
#         # Normalize image to [0, 1]
#         image = image.astype(np.float32) / 255.0
#
#         # Apply transform if provided (typically EventCountNormalization)
#         if self.transform is not None:
#             # Combine for transform: (H, W, C) where C = [pos_events, neg_events, image]
#             combined = np.concatenate([
#                 events.transpose(1, 2, 0),  # (H, W, 2)
#                 image[..., np.newaxis]       # (H, W, 1)
#             ], axis=-1)
#
#             combined = self.transform(combined)
#
#             # Split back
#             events = combined[..., :-1].transpose(2, 0, 1)  # (2, H, W)
#             image = combined[..., -1]                        # (H, W)
#
#         return (
#             torch.from_numpy(events.astype(np.float32)).contiguous(),
#             torch.from_numpy(image[np.newaxis].astype(np.float32)).contiguous(),
#             torch.tensor(1.0, dtype=torch.float32)
#         )
#
#
# class MemmapSyntheticDataset(Dataset):
#     """
#     Memory-mapped dataset loader (FASTEST for training).
#     Loads from events_all.npy and images_all.npy.
#     """
#
#     def __init__(self, events_file, images_file, transform=None):
#         """
#         Args:
#             events_file: Path to events_all.npy (N, 2, H, W)
#             images_file: Path to images_all.npy (N, H, W)
#             transform: Optional transform
#         """
#         # transform=None
#         self.transform = transform
#
#         # Memory-map the arrays (doesn't load into RAM)
#         self.events_mmap = np.load(events_file, mmap_mode='r')
#         self.images_mmap = np.load(images_file, mmap_mode='r')
#
#         assert len(self.events_mmap) == len(self.images_mmap), \
#             f"Events ({len(self.events_mmap)}) and images ({len(self.images_mmap)}) count mismatch"
#
#         print(f"Loaded memory-mapped dataset: {len(self.events_mmap)} samples")
#         print(f"  Events: {events_file} - {self.events_mmap.shape}")
#         print(f"  Images: {images_file} - {self.images_mmap.shape}")
#
#     def __len__(self):
#         return len(self.events_mmap)
#
#     def __getitem__(self, idx):
#         """
#         Load sample from memory-mapped arrays.
#
#         Returns:
#             events: (2, H, W) tensor
#             target: (1, H, W) tensor
#             integration_time: scalar tensor
#         """
#         # Access memory-mapped data (OS caches automatically)
#         events = self.events_mmap[idx].astype(np.float32)
#         image = self.images_mmap[idx].astype(np.float32)
#
#         # Apply transform if provided
#         if self.transform is not None:
#             combined = np.concatenate([
#                 events.transpose(1, 2, 0),
#                 image[..., np.newaxis]
#             ], axis=-1)
#
#             combined = self.transform(combined)
#
#             events = combined[..., :-1].transpose(2, 0, 1)
#             image = combined[..., -1]
#
#         return (
#             torch.from_numpy(events.copy()).contiguous(),
#             torch.from_numpy(image[np.newaxis].copy()).contiguous(),
#             torch.tensor(1.0, dtype=torch.float32)
#         )
#
#
# class HDF5SyntheticDataset(Dataset):
#     """
#     HDF5 dataset loader (good balance of speed and compression).
#     Loads from synthetic_dataset.h5.
#     """
#
#     def __init__(self, h5_file, transform=None):
#         """
#         Args:
#             h5_file: Path to .h5 file containing 'events' and 'images' datasets
#             transform: Optional transform
#         """
#         import h5py
#
#         self.h5_file = h5_file
#         self.transform = transform
#
#         # Open HDF5 file (keep handle open for fast access)
#         self.h5 = h5py.File(h5_file, 'r')
#         self.events_ds = self.h5['events']
#         self.images_ds = self.h5['images']
#
#         assert len(self.events_ds) == len(self.images_ds), \
#             f"Events ({len(self.events_ds)}) and images ({len(self.images_ds)}) count mismatch"
#
#         print(f"Loaded HDF5 dataset: {len(self.events_ds)} samples")
#         print(f"  File: {h5_file}")
#         print(f"  Events shape: {self.events_ds.shape}")
#         print(f"  Images shape: {self.images_ds.shape}")
#
#     def __len__(self):
#         return len(self.events_ds)
#
#     def __getitem__(self, idx):
#         """
#         Load sample from HDF5 file.
#
#         Returns:
#             events: (2, H, W) tensor
#             target: (1, H, W) tensor
#             integration_time: scalar tensor
#         """
#         # Load from HDF5 (decompresses on-the-fly)
#         events = self.events_ds[idx].astype(np.float32)
#         image = self.images_ds[idx].astype(np.float32)
#
#         # Apply transform if provided
#         if self.transform is not None:
#             combined = np.concatenate([
#                 events.transpose(1, 2, 0),
#                 image[..., np.newaxis]
#             ], axis=-1)
#
#             combined = self.transform(combined)
#
#             events = combined[..., :-1].transpose(2, 0, 1)
#             image = combined[..., -1]
#
#         return (
#             torch.from_numpy(events.copy()).contiguous(),
#             torch.from_numpy(image[np.newaxis].copy()).contiguous(),
#             torch.tensor(1.0, dtype=torch.float32)
#         )
#
#     def __del__(self):
#         """Close HDF5 file on cleanup."""
#         if hasattr(self, 'h5'):
#             self.h5.close()
#
#
# class PrecomputedSyntheticDatasetFast(Dataset):
#     """
#     Fast version that skips transforms and assumes events already normalized.
#     Works with individual .npy files.
#     """
#
#     def __init__(self, events_folder, images_folder):
#         """
#         Args:
#             events_folder: Folder containing .npy event files (already normalized)
#             images_folder: Folder containing .png image files
#         """
#         self.events_folder = Path(events_folder)
#         self.images_folder = Path(images_folder)
#
#         # Get all event files (.npy)
#         event_files = sorted(self.events_folder.glob('*.npy'))
#
#         # Filter out memory-mapped files
#         event_files = [f for f in event_files if f.stem not in ['events_all', 'images_all']]
#
#         # Verify matching image files exist
#         self.sample_names = []
#         for event_file in event_files:
#             stem = event_file.stem
#             image_file = self.images_folder / f"{stem}.png"
#
#             if image_file.exists():
#                 self.sample_names.append(stem)
#             else:
#                 print(f"Warning: No matching image for {event_file.name}")
#
#         print(f"Loaded {len(self.sample_names)} precomputed samples (fast mode)")
#         print(f"  Events: {self.events_folder}")
#         print(f"  Images: {self.images_folder}")
#
#     def __len__(self):
#         return len(self.sample_names)
#
#     def __getitem__(self, idx):
#         """
#         Load precomputed event and image pair (no transforms).
#
#         Returns:
#             events: (2, H, W) tensor of normalized event counts
#             target: (1, H, W) tensor of target image [0, 1]
#             integration_time: scalar tensor (1.0)
#         """
#         sample_name = self.sample_names[idx]
#
#         # Load events from .npy (already normalized)
#         event_path = self.events_folder / f"{sample_name}.npy"
#         events = np.load(event_path).astype(np.float32)
#
#         # Ensure contiguous memory layout for efficient GPU transfer
#         if not events.flags['C_CONTIGUOUS']:
#             events = np.ascontiguousarray(events)
#
#         # Load image from .png
#         image_path = self.images_folder / f"{sample_name}.png"
#         image = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
#
#         return (
#             torch.from_numpy(events).contiguous(),
#             torch.from_numpy(image[np.newaxis]).contiguous(),
#             torch.tensor(1.0, dtype=torch.float32)
#         )
#
#
# def auto_detect_format(data_folder):
#     """
#     Auto-detect which format the dataset is in.
#
#     Returns:
#         'memmap', 'hdf5', or 'npy'
#     """
#     folder = Path(data_folder)
#
#     # Check for memory-mapped files
#     if (folder / 'synth_events' / 'events_all.npy').exists() and \
#        (folder / 'synth_images' / 'images_all.npy').exists():
#         return 'memmap'
#
#     # Check for HDF5 file
#     if (folder / 'synthetic_dataset.h5').exists():
#         return 'hdf5'
#
#     # Check for individual .npy files
#     events_folder = folder / 'synth_events'
#     if events_folder.exists() and len(list(events_folder.glob('*.npy'))) > 0:
#         return 'npy'
#
#     raise ValueError(f"Could not detect dataset format in {data_folder}")
#
#
# def load_dataset(data_folder, format='auto', transform=None, fast_mode=False):
#     """
#     Convenience function to load dataset in any format.
#
#     Args:
#         data_folder: Root folder containing dataset
#         format: 'auto', 'memmap', 'hdf5', or 'npy'
#         transform: Optional transform
#         fast_mode: Skip transforms (only for 'npy' format)
#
#     Returns:
#         Dataset instance
#     """
#     folder = Path(data_folder)
#
#     if format == 'auto':
#         format = auto_detect_format(data_folder)
#         print(f"Auto-detected format: {format}")
#
#     if format == 'memmap':
#         events_file = folder / 'synth_events' / 'events_all.npy'
#         images_file = folder / 'synth_images' / 'images_all.npy'
#         return MemmapSyntheticDataset(events_file, images_file, transform=transform)
#
#     elif format == 'hdf5':
#         h5_file = folder / 'synthetic_dataset.h5'
#         return HDF5SyntheticDataset(h5_file, transform=transform)
#
#     elif format == 'npy':
#         events_folder = folder / 'synth_events'
#         images_folder = folder / 'synth_images'
#
#         if fast_mode:
#             return PrecomputedSyntheticDatasetFast(events_folder, images_folder)
#         else:
#             return PrecomputedSyntheticDataset(events_folder, images_folder, transform=transform)
#
#     else:
#         raise ValueError(f"Unknown format: {format}")
#
#
# class PrecomputedSyntheticDataset(Dataset):
#     """
#     Dataset that loads precomputed synthetic events and corresponding images.
#     All augmentations and preprocessing already done - just load and return.
#     """
#
#     def __init__(self, events_folder, images_folder, transform=None):
#         """
#         Args:
#             events_folder: Folder containing .npy event files (shape: 2, H, W)
#             images_folder: Folder containing .png image files
#             transform: Optional transform (typically just EventCountNormalization)
#         """
#         self.events_folder = Path(events_folder)
#         self.images_folder = Path(images_folder)
#         self.transform = transform
#
#         # Get all event files (.npy)
#         event_files = sorted(self.events_folder.glob('*.npy'))
#
#         # Verify matching image files exist
#         self.sample_names = []
#         for event_file in event_files:
#             stem = event_file.stem
#             image_file = self.images_folder / f"{stem}.png"
#
#             if image_file.exists():
#                 self.sample_names.append(stem)
#             else:
#                 print(f"Warning: No matching image for {event_file.name}")
#
#         print(f"Loaded {len(self.sample_names)} precomputed samples")
#         print(f"  Events: {self.events_folder}")
#         print(f"  Images: {self.images_folder}")
#
#     def __len__(self):
#         return len(self.sample_names)
#
#     def __getitem__(self, idx):
#         """
#         Load precomputed event and image pair.
#
#         Returns:
#             events: (C, H, W) tensor of synthetic event counts
#             target: (1, H, W) tensor of target image
#             integration_time: scalar tensor (always 1.0)
#         """
#         sample_name = self.sample_names[idx]
#
#         # Load events from .npy (single array file)
#         event_path = self.events_folder / f"{sample_name}.npy"
#         events = np.load(event_path)  # Directly returns array, shape: (2, H, W)
#
#         # Ensure float32 and contiguous
#         if events.dtype != np.float32:
#             events = events.astype(np.float32)
#         if not events.flags['C_CONTIGUOUS']:
#             events = np.ascontiguousarray(events)
#
#         # Load image from .png
#         image_path = self.images_folder / f"{sample_name}.png"
#         image = np.array(Image.open(image_path).convert('L'))  # Grayscale
#
#         # Normalize image to [0, 1]
#         image = image.astype(np.float32) / 255.0
#
#         # Apply transform if provided (typically EventCountNormalization)
#         if self.transform is not None:
#             # Combine for transform: (H, W, C) where C = [pos_events, neg_events, image]
#             combined = np.concatenate([
#                 events.transpose(1, 2, 0),  # (H, W, 2)
#                 image[..., np.newaxis]       # (H, W, 1)
#             ], axis=-1)
#
#             combined = self.transform(combined)
#
#             # Split back
#             events = combined[..., :-1].transpose(2, 0, 1)  # (2, H, W)
#             image = combined[..., -1]                        # (H, W)
#
#         return (
#             torch.from_numpy(events.astype(np.float32)).contiguous(),
#             torch.from_numpy(image[np.newaxis].astype(np.float32)).contiguous(),
#             torch.tensor(1.0, dtype=torch.float32)
#         )
#
#
# class PrecomputedSyntheticDatasetFast(Dataset):
#     """
#     Faster version that skips transform and assumes events are already normalized.
#     Use this if you pre-normalized events during database creation.
#     """
#
#     def __init__(self, events_folder, images_folder):
#         """
#         Args:
#             events_folder: Folder containing .npy event files (already normalized)
#             images_folder: Folder containing .png image files
#         """
#         self.events_folder = Path(events_folder)
#         self.images_folder = Path(images_folder)
#
#         # Get all event files (.npy)
#         event_files = sorted(self.events_folder.glob('*.npy'))
#
#         # Verify matching image files exist
#         self.sample_names = []
#         for event_file in event_files:
#             stem = event_file.stem
#             image_file = self.images_folder / f"{stem}.png"
#
#             if image_file.exists():
#                 self.sample_names.append(stem)
#             else:
#                 print(f"Warning: No matching image for {event_file.name}")
#
#         print(f"Loaded {len(self.sample_names)} precomputed samples (fast mode)")
#         print(f"  Events: {self.events_folder}")
#         print(f"  Images: {self.images_folder}")
#
#     def __len__(self):
#         return len(self.sample_names)
#
#     def __getitem__(self, idx):
#         """
#         Load precomputed event and image pair (no transforms).
#
#         Returns:
#             events: (2, H, W) tensor of normalized event counts
#             target: (1, H, W) tensor of target image [0, 1]
#             integration_time: scalar tensor (1.0)
#         """
#         sample_name = self.sample_names[idx]
#
#         # Load events from .npy (already normalized)
#         event_path = self.events_folder / f"{sample_name}.npy"
#         events = np.load(event_path).astype(np.float32)  # Directly returns array
#
#         # Load image from .png
#         image_path = self.images_folder / f"{sample_name}.png"
#         # image = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
#         image = np.array(Image.open(image_path).convert('L')).astype(np.float32)
#
#         return (
#             torch.from_numpy(events),
#             torch.from_numpy(image[np.newaxis]),
#             torch.tensor(1.0, dtype=torch.float32)
#         )

"""Dataset loaders for precomputed synthetic events and images - supports multiple formats."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def bin_array_reduceat(arr, bin_size):
    """
    Bin a 2D array by summing values in bins of specified size.
    Most efficient for regular binning.
    """
    if arr.shape[0] % bin_size != 0 or arr.shape[1] % bin_size != 0:
        # Trim array to make it divisible by bin_size
        new_h = (arr.shape[0] // bin_size) * bin_size
        new_w = (arr.shape[1] // bin_size) * bin_size
        arr = arr[:new_h, :new_w]

    # Create indices for reduceat
    indices_h = np.arange(0, arr.shape[0], bin_size)
    indices_w = np.arange(0, arr.shape[1], bin_size)

    # Apply reduceat along both dimensions
    temp = np.add.reduceat(arr, indices_h, axis=0)
    result = np.add.reduceat(temp, indices_w, axis=1)

    # return result
    return result / (bin_size * bin_size)


class BinomialSampling(object):
    """
    Sample event counts from binomial distribution given event probabilities.

    Input: (H, W, C) array where C contains [pos_prob, neg_prob, image]
    Output: (H, W, C) array where C contains [pos_counts, neg_counts, image]
    """

    def __init__(self, time_steps=1e6, bin_size=2):
        self.time_steps = time_steps
        self.bin_size =  bin_size

    def __call__(self, prob_array):
        if not isinstance(prob_array, np.ndarray):
            raise TypeError(f'Input should be numpy array. Got {type(prob_array)}')

        if prob_array.ndim != 3:
            raise ValueError(f'Input should be 3D (H, W, C). Got shape {prob_array.shape}')

        if prob_array.shape[2] < 3:
            raise ValueError(f'Input should have at least 3 channels. Got {prob_array.shape[2]}')

        # #compute the effective time steps
        m_pos = prob_array[:, :, 0]
        m_neg = prob_array[:, :, 1]
        image = prob_array[:, :, 2]
        # time_steps_array = np.zeros_like(m_pos)
        # time_steps_array = np.where(time_steps_array == 0, self.time_steps, time_steps_array)

        sampled_events_pos = np.random.binomial(
            self.time_steps,
            m_pos
        ).astype(np.float32)

        sampled_events_neg = np.random.binomial(
            self.time_steps,
            m_neg
        ).astype(np.float32)

        if self.bin_size > 1:
            sampled_events_pos = bin_array_reduceat(sampled_events_pos, bin_size=self.bin_size).astype(np.float32)
            sampled_events_neg = bin_array_reduceat(sampled_events_neg, bin_size=self.bin_size).astype(np.float32)
            image = bin_array_reduceat(image, bin_size=self.bin_size)

        events = np.stack(
            [sampled_events_pos, sampled_events_neg, image],
            axis=2
        ).astype(np.float32)

        # sampled_events=sampled_events_pos+sampled_events_neg
        # sampled_events*=10
        # image*=255
        #
        # plt.imshow(sampled_events, cmap='hot', vmin=0, vmax=255)
        # plt.show()
        # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        # plt.show()


        return events


class MemmapSyntheticDataset(Dataset):
    """
    Memory-mapped dataset loader (FASTEST for training).
    Loads from events_all.npy and images_all.npy.
    """

    def __init__(self, events_file, images_file, transform=None, preload_to_ram=True):
        """
        Args:
            events_file: Path to events_all.npy (N, 2, H, W)
            images_file: Path to images_all.npy (N, H, W)
            transform: Optional transform
            preload_to_ram: If True, loads entire dataset to RAM (recommended for random access)
        """
        self.transform = transform
        self.preload_to_ram = preload_to_ram

        if preload_to_ram:
            print(f"Preloading dataset to RAM (this may take 30-60 seconds)...")
            # Load entire arrays into RAM
            self.events_data = np.load(events_file, mmap_mode='r')[:]  # [:] copies to RAM
            self.images_data = np.load(images_file, mmap_mode='r')[:]
            print(f"Preloading complete! Using {self.events_data.nbytes/1e9:.2f}GB + {self.images_data.nbytes/1e9:.2f}GB RAM")
        else:
            # Memory-map only (fast for sequential access, slow for random)
            print("Using memory-mapped mode (fast for sequential, slow for shuffle)")
            self.events_data = np.load(events_file, mmap_mode='r')
            self.images_data = np.load(images_file, mmap_mode='r')

        assert len(self.events_data) == len(self.images_data), \
            f"Events ({len(self.events_data)}) and images ({len(self.images_data)}) count mismatch"

        print(f"Loaded dataset: {len(self.events_data)} samples")
        print(f"  Events: {events_file} - {self.events_data.shape}")
        print(f"  Images: {images_file} - {self.images_data.shape}")
        if preload_to_ram:
            print(f"  Mode: RAM (fast random access, shuffle friendly)")
        else:
            print(f"  Mode: Memory-mapped (use shuffle=False for best performance)")

    def __len__(self):
        return len(self.events_data)

    def __getitem__(self, idx):
        """
        Load sample from memory-mapped arrays.

        Returns:
            events: (2, H, W) tensor
            target: (1, H, W) tensor
            integration_time: scalar tensor
        """
        # Access data (RAM or mmap)
        events = self.events_data[idx]
        image = self.images_data[idx]

        # Apply transform if provided
        if self.transform is not None:
            events = events.astype(np.float16, copy=True)
            image = image.astype(np.float16, copy=True)

            combined = np.concatenate([
                events.transpose(1, 2, 0),
                image[..., np.newaxis]
            ], axis=-1)

            combined = self.transform(combined)

            events = combined[..., :-1].transpose(2, 0, 1)
            image = combined[..., -1]

            # Ensure arrays are contiguous after transforms (flips create negative strides)
            if not events.flags['C_CONTIGUOUS']:
                events = np.ascontiguousarray(events)
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)

            return (
                torch.from_numpy(events),
                torch.from_numpy(image[np.newaxis]),
                torch.tensor(1.0, dtype=torch.float16)
            )
        else:
            # Fast path: ensure arrays are writable to avoid UB
            events = np.asarray(events, dtype=np.float32)
            image = np.asarray(image, dtype=np.float32)

            if not events.flags.writeable:
                events = events.copy()
            if not image.flags.writeable:
                image = image.copy()

            return (
                torch.from_numpy(events),
                torch.from_numpy(image[np.newaxis]),
                torch.tensor(1.0, dtype=torch.float32)
            )


class HDF5SyntheticDataset(Dataset):
    """
    HDF5 dataset loader (good balance of speed and compression).
    Loads from synthetic_dataset.h5.
    """

    def __init__(self, h5_file, transform=None):
        """
        Args:
            h5_file: Path to .h5 file containing 'events' and 'images' datasets
            transform: Optional transform
        """
        import h5py

        self.h5_file = h5_file
        self.transform = transform

        # Open HDF5 file (keep handle open for fast access)
        self.h5 = h5py.File(h5_file, 'r')
        self.events_ds = self.h5['events']
        self.images_ds = self.h5['images']

        assert len(self.events_ds) == len(self.images_ds), \
            f"Events ({len(self.events_ds)}) and images ({len(self.images_ds)}) count mismatch"

        print(f"Loaded HDF5 dataset: {len(self.events_ds)} samples")
        print(f"  File: {h5_file}")
        print(f"  Events shape: {self.events_ds.shape}")
        print(f"  Images shape: {self.images_ds.shape}")

    def __len__(self):
        return len(self.events_ds)

    def __getitem__(self, idx):
        """
        Load sample from HDF5 file.

        Returns:
            events: (2, H, W) tensor
            target: (1, H, W) tensor
            integration_time: scalar tensor
        """
        # Load from HDF5 (decompresses on-the-fly)
        events = self.events_ds[idx].astype(np.float32)
        image = self.images_ds[idx].astype(np.float32)

        # Apply transform if provided
        if self.transform is not None:
            combined = np.concatenate([
                events.transpose(1, 2, 0),
                image[..., np.newaxis]
            ], axis=-1)

            combined = self.transform(combined)

            events = combined[..., :-1].transpose(2, 0, 1)
            image = combined[..., -1]

        return (
            torch.from_numpy(events.copy()).contiguous(),
            torch.from_numpy(image[np.newaxis].copy()).contiguous(),
            torch.tensor(1.0, dtype=torch.float32)
        )

    def __del__(self):
        """Close HDF5 file on cleanup."""
        if hasattr(self, 'h5'):
            self.h5.close()


def auto_detect_format(data_folder):
    """
    Auto-detect which format the dataset is in.

    Returns:
        'memmap', 'hdf5', or 'npy'
    """
    folder = Path(data_folder)

    # Check for memory-mapped files
    if (folder / 'synth_events' / 'events_all.npy').exists() and \
       (folder / 'synth_images' / 'images_all.npy').exists():
        return 'memmap'

    # Check for HDF5 file
    if (folder / 'synthetic_dataset.h5').exists():
        return 'hdf5'

    # Check for individual .npy files
    events_folder = folder / 'synth_events'
    if events_folder.exists() and len(list(events_folder.glob('*.npy'))) > 0:
        return 'npy'

    raise ValueError(f"Could not detect dataset format in {data_folder}")


def load_dataset(data_folder, format='auto', transform=None, fast_mode=False, preload_to_ram=True):
    """
    Convenience function to load dataset in any format.

    Args:
        data_folder: Root folder containing dataset
        format: 'auto', 'memmap', 'hdf5', or 'npy'
        transform: Optional transform
        fast_mode: Skip transforms (only for 'npy' format)
        preload_to_ram: For memmap format, load entire dataset to RAM (enables fast shuffle)

    Returns:
        Dataset instance
    """
    folder = Path(data_folder)

    if format == 'auto':
        format = auto_detect_format(data_folder)
        print(f"Auto-detected format: {format}")

    if format == 'memmap':
        events_file = folder / 'synth_events' / 'events_all.npy'
        images_file = folder / 'synth_images' / 'images_all.npy'
        return MemmapSyntheticDataset(events_file, images_file, transform=transform,
                                     preload_to_ram=preload_to_ram)

    elif format == 'hdf5':
        h5_file = folder / 'synthetic_dataset.h5'
        return HDF5SyntheticDataset(h5_file, transform=transform)

    elif format == 'npy':
        events_folder = folder / 'synth_events'
        images_folder = folder / 'synth_images'

        if fast_mode:
            return PrecomputedSyntheticDatasetFast(events_folder, images_folder)
        else:
            return PrecomputedSyntheticDataset(events_folder, images_folder, transform=transform)

    else:
        raise ValueError(f"Unknown format: {format}")


def preload_dataset_to_gpu(dataset, device='cuda'):
    """Load entire dataset to GPU memory - eliminates PCIe bottleneck."""
    print(f"Preloading {len(dataset)} samples to GPU VRAM...")

    all_events = []
    all_targets = []
    all_times = []

    for i in range(len(dataset)):
        if i % 100 == 0:
            print(f"  Loading {i}/{len(dataset)}...")

        events, target, int_time = dataset[i]
        all_events.append(events.to(device))
        all_targets.append(target.to(device))
        all_times.append(int_time.to(device))

    # Stack into tensors on GPU
    events_gpu = torch.stack(all_events)
    targets_gpu = torch.stack(all_targets)
    times_gpu = torch.stack(all_times)

    print(f"Dataset on GPU: {events_gpu.element_size() * events_gpu.nelement() / 1e9:.2f}GB")

    return torch.utils.data.TensorDataset(events_gpu, targets_gpu, times_gpu)


class PrecomputedSyntheticDataset(Dataset):
    """
    Dataset that loads precomputed synthetic events and corresponding images.
    All augmentations and preprocessing already done - just load and return.
    """

    def __init__(self, events_folder, images_folder, transform=None):
        """
        Args:
            events_folder: Folder containing .npy event files (shape: 2, H, W)
            images_folder: Folder containing .png image files
            transform: Optional transform (typically just EventCountNormalization)
        """
        self.events_folder = Path(events_folder)
        self.images_folder = Path(images_folder)
        self.transform = transform

        # Get all event files (.npy)
        event_files = sorted(self.events_folder.glob('*.npy'))

        # Verify matching image files exist
        self.sample_names = []
        for event_file in event_files:
            stem = event_file.stem
            image_file = self.images_folder / f"{stem}.png"

            if image_file.exists():
                self.sample_names.append(stem)
            else:
                print(f"Warning: No matching image for {event_file.name}")

        print(f"Loaded {len(self.sample_names)} precomputed samples")
        print(f"  Events: {self.events_folder}")
        print(f"  Images: {self.images_folder}")

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        """
        Load precomputed event and image pair.

        Returns:
            events: (C, H, W) tensor of synthetic event counts
            target: (1, H, W) tensor of target image
            integration_time: scalar tensor (always 1.0)
        """
        sample_name = self.sample_names[idx]

        # Load events from .npy (single array file)
        event_path = self.events_folder / f"{sample_name}.npy"
        events = np.load(event_path)  # Directly returns array, shape: (2, H, W)

        # Ensure float32 and contiguous
        if events.dtype != np.float32:
            events = events.astype(np.float32)
        if not events.flags['C_CONTIGUOUS']:
            events = np.ascontiguousarray(events)

        # Load image from .png
        image_path = self.images_folder / f"{sample_name}.png"
        image = np.array(Image.open(image_path).convert('L'))  # Grayscale

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply transform if provided (typically EventCountNormalization)
        if self.transform is not None:
            # Combine for transform: (H, W, C) where C = [pos_events, neg_events, image]
            combined = np.concatenate([
                events.transpose(1, 2, 0),  # (H, W, 2)
                image[..., np.newaxis]       # (H, W, 1)
            ], axis=-1)

            combined = self.transform(combined)

            # Split back
            events = combined[..., :-1].transpose(2, 0, 1)  # (2, H, W)
            image = combined[..., -1]                        # (H, W)

        return (
            torch.from_numpy(events.astype(np.float32)).contiguous(),
            torch.from_numpy(image[np.newaxis].astype(np.float32)).contiguous(),
            torch.tensor(1.0, dtype=torch.float32)
        )


class PrecomputedSyntheticDatasetFast(Dataset):
    """
    Faster version that skips transform and assumes events are already normalized.
    Use this if you pre-normalized events during database creation.
    """

    def __init__(self, events_folder, images_folder):
        """
        Args:
            events_folder: Folder containing .npy event files (already normalized)
            images_folder: Folder containing .png image files
        """
        self.events_folder = Path(events_folder)
        self.images_folder = Path(images_folder)

        # Get all event files (.npy)
        event_files = sorted(self.events_folder.glob('*.npy'))

        # Verify matching image files exist
        self.sample_names = []
        for event_file in event_files:
            stem = event_file.stem
            image_file = self.images_folder / f"{stem}.png"

            if image_file.exists():
                self.sample_names.append(stem)
            else:
                print(f"Warning: No matching image for {event_file.name}")

        print(f"Loaded {len(self.sample_names)} precomputed samples (fast mode)")
        print(f"  Events: {self.events_folder}")
        print(f"  Images: {self.images_folder}")

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        """
        Load precomputed event and image pair (no transforms).

        Returns:
            events: (2, H, W) tensor of normalized event counts
            target: (1, H, W) tensor of target image [0, 1]
            integration_time: scalar tensor (1.0)
        """
        sample_name = self.sample_names[idx]

        # Load events from .npy (already normalized)
        event_path = self.events_folder / f"{sample_name}.npy"
        events = np.load(event_path).astype(np.float32)  # Directly returns array

        # Load image from .png
        image_path = self.images_folder / f"{sample_name}.png"
        image = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0

        return (
            torch.from_numpy(events),
            torch.from_numpy(image[np.newaxis]),
            torch.tensor(1.0, dtype=torch.float32)
        )