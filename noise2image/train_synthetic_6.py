
"""
train_synthetic_6.py — Train the reconstruction CNN on a precomputed
(events, image) pair database.

This is the sole training entry point.  The name is historical: it
now handles training on both synthetic and real (experimentally
recorded) noise-image pairs — see
:func:`noise2params.noise_image.real_data_base_maker` for the real-pair
construction.  Both paths produce the same precomputed-array layout
(``events_all.npy`` + ``images_all.npy``) that
:mod:`noise2image.precomputed_dataset` expects.  See the "Naming
conventions" section of the repository README.

Supports three storage formats for the dataset: ``memmap`` (the
default and fastest: raw ``.npy`` files memory-mapped by
:class:`numpy.memmap`), ``hdf5``, and ``npy`` (one file per sample).

At the top of this file there is an SDPA-fallback patch for the
``denoising_diffusion_pytorch`` Attend / Attention classes.  On some
Windows configurations the flash-attention kernel either is not
available or crashes; the patch replaces the flash_attn method on
those classes with a scoped
:func:`torch.nn.functional.scaled_dot_product_attention` call that
forces the math / mem-efficient kernels.  Leave this patch in place;
removing it will cause the first forward pass to abort on the
affected systems.

See the repository ``README.md`` for critical caveats about training:
    * Use ``--batch_size 2``.  Larger batch sizes have been observed to
      cause multi-order-of-magnitude training slowdowns on our GPUs.
    * Environment pinning is fragile; see the
      ``denoising-diffusion-pytorch==1.10.7`` / ``torchmetrics<1.8.0``
      remedy documented there.
"""



# --- SDPA safe fallback for lucidrains-style attend modules on Windows / no-FA wheels
import torch
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel

# Ensure global fallbacks are available
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# Import the offending module and patch its class method to allow math fallback
import denoising_diffusion_pytorch.attend as _ddp_attend

def _safe_flash_attn_method(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    # Force a permissive context so PyTorch always has at least the math kernel
    with sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True):
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )

# Handle both common class names used by lucidrains repos
for _cls_name in ("Attend", "Attention"):
    _cls = getattr(_ddp_attend, _cls_name, None)
    if _cls is not None and hasattr(_cls, "flash_attn"):
        setattr(_cls, "flash_attn", _safe_flash_attn_method)

# Confirm the patched method is in place
print("Patched Attend.flash_attn:",
      hasattr(getattr(_ddp_attend, "Attend", object), "flash_attn"))

# Minimal SDPA call under the same context
q = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
with sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True):
    y = F.scaled_dot_product_attention(q, k, v)
print("SDPA smoke test:", y.is_cuda, y.shape)

import inspect
print("Attend module file:", inspect.getsourcefile(_ddp_attend))
print("Available classes:", [n for n in dir(_ddp_attend) if n[0].isupper()])

from argparse import ArgumentParser
import os
import time

import numpy as np

from torch.utils.data import DataLoader
from lightning.pytorch import loggers
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision import transforms

from train import Model, MultiDataloaderEarlyStopping
import utils
from precomputed_dataset import load_dataset, preload_dataset_to_gpu, BinomialSampling
import np_transforms

torch.set_float32_matmul_precision('high')

parser = ArgumentParser()
parser.add_argument("--format", type=str, default='memmap',
                    choices=['auto', 'memmap', 'hdf5', 'npy'],
                    help="Dataset format: auto-detect, memmap (fastest), hdf5, or npy")
parser.add_argument("--gpu_ind", type=int, default=0, help="GPU index")
parser.add_argument("--vanilla_unet", action='store_true',
                    help="Use vanilla U-Net instead of the advanced u-net with attention layers")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for data loader")
parser.add_argument("--log_name", type=str, default='precomputed_training',
                    help="Name of the log & checkpoint folder")

parser.add_argument("--data_folder", type=str, default='./data/',
                    help="Root folder containing synth_events/ and synth_images/ for TRAINING")
parser.add_argument("--val_data_folders", type=str, nargs='+', default=None,
                    help="List of validation data folders (real event recordings). The first validation set given is used for checkpointing.")

parser.add_argument("--val_split", type=float, default=0.15,
                    help="Validation split ratio (only used if --val_data_folder not specified)")
parser.add_argument("--normalize_events", action='store_true',
                    help="Apply EventCountNormalization (use if events not pre-normalized)")
parser.add_argument("--fast_mode", action='store_true',
                    help="Skip transforms entirely (assumes events already normalized)")
parser.add_argument("--on_fly_sampling", action='store_true',
                    help="Samples events on the fly at each epoch. Must be used with precomputed probabilities.")
parser.add_argument("--integration_time", type=int, default=1,
                    help="Integration time for sampling events and normalization")
parser.add_argument("--pixel_bin", type=int, default=2,
                    help="Size of pixel bin to use. Only applicable if using on fly sampling.")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Path to checkpoint file to resume training from")

# Performance optimization flags
parser.add_argument("--mixed_precision", action='store_true',
                    help="Use mixed precision (fp16) - only beneficial with batch_size >= 8")
parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                    help="Gradient accumulation steps")
parser.add_argument("--prefetch_factor", type=int, default=4,
                    help="Number of batches to prefetch per worker")
parser.add_argument("--no_preload", action='store_true',
                    help="Don't preload memmap to RAM (use for sequential access only, no shuffle)")

def manual_training_test(ds_gpu, batch_size=2, num_iterations=10):
    # Get one batch
    loader = torch.utils.data.DataLoader(ds_gpu, batch_size=2)
    x, y, t = next(iter(loader))

    print(f"\n{'='*60}")
    print("ISOLATION TEST")
    print(f"{'='*60}")
    print(f"Data device: {x.device}")
    print(f"Data shape: {x.shape}")

    # Test 1: Just data loading (no model)
    print("\n[Test 1] Data loading only...")
    torch.cuda.synchronize()
    start = time.time()
    for i, (x, y, t) in enumerate(loader):
        if i >= 100:
            break
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  Speed: {100/elapsed:.2f} it/s")
    print(f"  Should be >1000 it/s since data already on GPU!")

    # Test 2: Dummy model (just convolution)
    print("\n[Test 2] Simple Conv2d model...")
    simple_model = torch.nn.Conv2d(2, 1, 3, padding=1).cuda()
    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        with torch.no_grad():
            out = simple_model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  Speed: {100/elapsed:.2f} it/s")
    print(f"  Should be >100 it/s for simple conv!")

    # Test 3: Full U-Net model
    print("\n[Test 3] Full U-Net model...")
    from train import Model
    full_model = Model(dim=64, in_channels=2, lr=5e-5, vanilla_unet=False).cuda()

    torch.cuda.synchronize()
    start = time.time()
    for i in range(10):  # Only 10 iterations for slow model
        with torch.no_grad():
            out = full_model(x, t)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  Speed: {10/elapsed:.2f} it/s")
    print(f"  Time per forward pass: {elapsed/10*1000:.1f}ms")

    # Test 4: Check for CPU fallback
    print("\n[Test 4] Checking for CPU operations...")
    torch.cuda.synchronize()

    # Profile one forward pass
    import torch.autograd.profiler as profiler
    with profiler.profile(use_cuda=True) as prof:
        out = full_model(x, t)

    # Check for CPU operations
    cpu_ops = [evt for evt in prof.key_averages() if evt.device_type == profiler.DeviceType.CPU]
    if cpu_ops:
        print(f"  ⚠️ Found {len(cpu_ops)} CPU operations!")
        for op in cpu_ops[:10]:  # Show first 10
            print(f"    - {op.key}: {op.cpu_time_total/1000:.2f}ms")
    else:
        print(f"  ✓ All operations on GPU")

if __name__ == '__main__':
    import torchmetrics, inspect
    from torchmetrics.image import PeakSignalNoiseRatio

    # print("torchmetrics:", torchmetrics.__version__)
    # print("PSNR signature:", inspect.signature(PeakSignalNoiseRatio.__init__))

    b = torch.backends.cuda
    print("SDPA backends -> flash:", b.flash_sdp_enabled(),
          "mem:", b.mem_efficient_sdp_enabled(),
          "cudnn:", b.cudnn_sdp_enabled(),
          "math:", b.math_sdp_enabled())

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    # Enable CUDA optimizations
    if torch.cuda.is_available():
        # torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("Enabled CUDA optimizations (TF32, cuDNN benchmark)")

    print("\n" + "=" * 60)
    print("TRAINING ON PRECOMPUTED SYNTHETIC DATABASE")
    print("=" * 60)
    print(f"Training data folder: {args.data_folder}")
    print(f"Validation data folders: {args.val_data_folders if args.val_data_folders else 'Split from training'}")
    print(f"Format: {args.format}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Mixed precision: {args.mixed_precision}")

    # Warn about mixed precision with small batches
    if args.mixed_precision and args.batch_size < 8:
        print("\n  WARNING: Mixed precision with batch_size < 8 may be slower (emphasis on may, test on your hardware)")
        print("   Recommendation: Use batch_size >= 8 or disable --mixed_precision")

    print("="*60 + "\n")

    if args.on_fly_sampling==True and args.fast_mode==True:
        raise ValueError('Cannot use fast mode and on fly sampling simultaneously')

    # Setup transforms
    if args.fast_mode:
        print("Fast mode: No transforms")
        transform = None
    else:
        transform_list = []

        if args.on_fly_sampling:
            print('Using on the fly sampling. Ensure that dataset is of probabilities')
            transform_list.append(BinomialSampling(
                time_steps=args.integration_time * 1e6,
                bin_size=args.pixel_bin
            ))
            transform_list.append(utils.EventCountNormalization(
                integration_time_s=args.integration_time
            ))
        elif args.normalize_events:
            print("Applying EventCountNormalization")
            transform_list.append(utils.EventCountNormalization(
                integration_time_s=args.integration_time
            ))
        else:
            print("No normalization.")

        # Add random flips when not in fast mode
        print("Adding random flip augmentations")
        transform_list.append(np_transforms.RandomHorizontalFlip())
        transform_list.append(np_transforms.RandomVerticalFlip())

        # Compose all transforms
        transform = transforms.Compose(transform_list) if transform_list else None

    # Load training dataset (synthetic)
    print("Loading TRAINING dataset (synthetic)...")
    ds_train = load_dataset(
        data_folder=args.data_folder,
        format=args.format,
        transform=transform,
        fast_mode=args.fast_mode,
        preload_to_ram=False
    )

    # Load validation datasets
    if args.val_data_folders is not None:
        print(f"Loading {len(args.val_data_folders)} validation datasets...")
        val_loaders = []
        for i, val_folder in enumerate(args.val_data_folders):
            print(f"  [{i}] Loading from {val_folder}...")
            ds_val = load_dataset(
                data_folder=val_folder,
                format=args.format,
                transform=transform,
                fast_mode=args.fast_mode,
                preload_to_ram=False,
            )
            val_loader = DataLoader(
                ds_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                persistent_workers=True if args.num_workers > 0 else False,
                pin_memory=True,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                shuffle=False
            )
            val_loaders.append(val_loader)
            print(f"    Samples: {len(ds_val)}")
    else:
        print(f"No separate validation folder specified. Splitting training data with val_split={args.val_split}")
        ds_train, ds_val, _ = utils.data_split(ds_train, validation_split=args.val_split,
                                               testing_split=0.0, seed=47)
        val_loader = DataLoader(
            ds_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=True if args.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            shuffle=False
        )
        val_loaders = [val_loader]

    # Preload to GPU instead
    # ds_gpu = preload_dataset_to_gpu(ds, device='cuda')

    # manual_training_test(ds_gpu)

    # # Split AFTER preloading
    # ds_train, ds_val, _ = utils.data_split(ds_gpu, validation_split=args.val_split,
    #                                        testing_split=0.0, seed=47)

    print(f"Training samples: {len(ds_train)}")
    if args.val_data_folders is not None:
        print(f"Total validation samples: {sum(len(loader.dataset) for loader in val_loaders)}")
    else:
        print(f"Validation samples: {len(ds_val)}")

    # Check sample to get dimensions
    sample_events, sample_image, _ = ds_train[0]
    in_channels = sample_events.shape[0]
    print(f"Event shape: {sample_events.shape}")
    print(f"Image shape: {sample_image.shape}")
    print(f"Input channels: {in_channels}")

    # Create optimized dataloaders
    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True, #default
        # pin_memory=False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        # shuffle=True
    )

    # Initialize model
    model = Model(dim=64, in_channels=in_channels, lr=args.lr,
                  vanilla_unet=args.vanilla_unet)

    # Enable channels-last for conv-heavy networks on Ada/Blackwell
    model = model.to(memory_format=torch.channels_last)
    print("Model converted to channels-last memory format")

    # Compile model for 5-20% speedup on PyTorch 2.x
    # Skip on Windows due to Triton compatibility issues
    import sys

    if sys.platform != 'win32':
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            print("Model compiled with torch.compile (max-autotune mode)")
        except Exception as e:
            print(f"torch.compile not available: {e}")
    else:
        print("torch.compile skipped on Windows (Triton not supported)")

    # Setup logger
    tb_logger = loggers.TensorBoardLogger('lightning_logs', name=args.log_name)

    # Configure trainer
    trainer_kwargs = {
        'logger': tb_logger,
        'callbacks': [
            ModelCheckpoint(monitor='val_loss/dataloader_idx_0', save_top_k=3, save_last=True,
                            mode='min', every_n_epochs=1),
            LearningRateMonitor(logging_interval='epoch',),
            MultiDataloaderEarlyStopping(patience=5, verbose=True),
        ],
        'accelerator': 'gpu',
        'devices': [args.gpu_ind],
        'max_epochs': args.num_epochs,
        'accumulate_grad_batches': args.accumulate_grad_batches,
    }

    # Add mixed precision if requested
    if args.mixed_precision:
        trainer_kwargs['precision'] = '16-mixed'
        # trainer_kwargs['precision'] = 'bf16-mixed'
        print("Using mixed precision (fp16) training")
        # print("Using mixed precision (bf16) training")

    trainer = Trainer(**trainer_kwargs)

    # Train
    print(f"\nStarting training...")
    print(f"Effective batch size: {args.batch_size * args.accumulate_grad_batches}")
    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loaders, ckpt_path=args.resume_from)

    # Test on validation set
    trainer.test(model, val_loaders)

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = trainer.predict(model, dataloaders=val_loaders)

    # Save predictions per dataloader
    if len(val_loaders) > 1:
        # predictions is a list of lists, one per dataloader
        for dl_idx, dl_predictions in enumerate(predictions):
            save_dict = {'pred': np.concatenate(dl_predictions)}
            save_path = os.path.join(tb_logger.log_dir, f'predictions_dl{dl_idx}.npz')
            np.savez(save_path, **save_dict)
            print(f"Predictions for dataloader {dl_idx} saved to {save_path}")
    else:
        # Single dataloader - predictions is already a flat list
        save_dict = {'pred': np.concatenate(predictions)}
        save_path = os.path.join(tb_logger.log_dir, 'predictions.npz')
        np.savez(save_path, **save_dict)
        print(f"Predictions saved to {save_path}")

    # Flatten nested lists and convert on Torch side
    # pred_t = torch.cat(
    #     [p.detach().to(torch.float32).cpu() for p in predictions], dim=0
    # ) if isinstance(predictions[0], torch.Tensor) or torch.is_tensor(predictions[0][0]) else \
    #     torch.cat(
    #         [pp.detach().to(torch.float32).cpu() for p in predictions for pp in
    #          (p if isinstance(p, (list, tuple)) else [p])],
    #         dim=0
    #     )
    # save_dict = {'pred': pred_t.numpy()}

    # save_path = os.path.join(tb_logger.log_dir, 'predictions.npz')
    # np.savez(save_path, **save_dict)
    # print(f"Predictions saved to {save_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)