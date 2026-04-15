"""
Bone segmentation inference using nnU-Net.

Runs nnUNetv2_predict on ear volumes prepared with nnU-Net naming convention
(from P4 preprocessing). Uses a 5-fold ensemble with best checkpoint.

Usage:
    python model/test_bone.py --input_dir /path/to/P4_Normalized_Ears_nnUNet --output_dir /path/to/output
    python model/test_bone.py --input_dir /path/to/input --output_dir /path/to/output --device cpu
    python model/test_bone.py --input_dir /path/to/input --output_dir /path/to/output --folds 0 1 2
"""

import os
import argparse
import subprocess
import sys


def get_default_device():
    """Check if a compatible GPU is available, otherwise default to cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if float(f"{major}.{minor}") >= 7.0:
                return "cuda"
            else:
                print(f"GPU compute capability {major}.{minor} < 7.0, defaulting to CPU")
                return "cpu"
    except ImportError:
        pass
    return "cpu"


def run_nnunet_predict(input_dir, output_dir, dataset_id=1, configuration="3d_fullres",
                       folds=None, checkpoint="checkpoint_best.pth", device=None,
                       trainer=None, plans=None,
                       nnunet_raw=None, nnunet_preprocessed=None, nnunet_results=None):
    """
    Run nnU-Net prediction.

    Args:
        input_dir: Directory with input images (*_0000.nii.gz)
        output_dir: Directory to save predictions
        dataset_id: nnU-Net dataset ID
        configuration: nnU-Net configuration (e.g., '3d_fullres')
        folds: List of fold numbers to use (default: [0,1,2,3,4])
        checkpoint: Checkpoint filename
        device: 'cpu' or 'cuda' (auto-detected if None)
        trainer: nnU-Net trainer class name (optional)
        plans: nnU-Net plans name (optional)
        nnunet_raw: Override nnUNet_raw env variable
        nnunet_preprocessed: Override nnUNet_preprocessed env variable
        nnunet_results: Override nnUNet_results env variable
    """
    if folds is None:
        folds = [0, 1, 2, 3, 4]
    if device is None:
        device = get_default_device()

    # Set environment variables if provided
    env = os.environ.copy()
    if nnunet_raw:
        env["nnUNet_raw"] = nnunet_raw
    if nnunet_preprocessed:
        env["nnUNet_preprocessed"] = nnunet_preprocessed
    if nnunet_results:
        env["nnUNet_results"] = nnunet_results

    # Build the command
    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", str(dataset_id),
        "-c", configuration,
        "-f", *[str(f) for f in folds],
        "-chk", checkpoint,
        "-device", device,
    ]

    if trainer:
        cmd.extend(["-tr", trainer])
    if plans:
        cmd.extend(["-p", plans])

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("BONE SEGMENTATION INFERENCE (nnU-Net)")
    print("=" * 80)
    print(f"  Input:         {input_dir}")
    print(f"  Output:        {output_dir}")
    print(f"  Dataset ID:    {dataset_id}")
    print(f"  Configuration: {configuration}")
    print(f"  Folds:         {folds}")
    print(f"  Checkpoint:    {checkpoint}")
    print(f"  Device:        {device}")
    if trainer:
        print(f"  Trainer:       {trainer}")
    if plans:
        print(f"  Plans:         {plans}")
    print("=" * 80 + "\n")

    # Count input files
    input_files = [f for f in os.listdir(input_dir) if f.endswith("_0000.nii.gz")]
    print(f"Found {len(input_files)} input cases\n")

    if len(input_files) == 0:
        print(f"ERROR: No *_0000.nii.gz files found in {input_dir}")
        sys.exit(1)

    # Run prediction
    print("Running nnUNetv2_predict...")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"\nERROR: nnUNetv2_predict exited with code {result.returncode}")
        sys.exit(result.returncode)

    print("\nBone segmentation inference completed.")
    print(f"Predictions saved to: {output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run bone segmentation inference using nnU-Net",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory with input images (*_0000.nii.gz)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save predictions")
    parser.add_argument("--dataset_id", type=int, default=1,
                        help="nnU-Net dataset ID (default: 1)")
    parser.add_argument("--configuration", default="3d_fullres",
                        help="nnU-Net configuration (default: 3d_fullres)")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Folds to use for ensemble (default: 0 1 2 3 4)")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth",
                        help="Checkpoint filename (default: checkpoint_best.pth)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="Device for inference (default: auto-detect)")
    parser.add_argument("--trainer", default=None,
                        help="nnU-Net trainer class name (optional)")
    parser.add_argument("--plans", default=None,
                        help="nnU-Net plans name (optional)")
    parser.add_argument("--nnunet_raw", default=None,
                        help="Override nnUNet_raw environment variable")
    parser.add_argument("--nnunet_preprocessed", default=None,
                        help="Override nnUNet_preprocessed environment variable")
    parser.add_argument("--nnunet_results", default=None,
                        help="Override nnUNet_results environment variable")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_nnunet_predict(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        folds=args.folds,
        checkpoint=args.checkpoint,
        device=args.device,
        trainer=args.trainer,
        plans=args.plans,
        nnunet_raw=args.nnunet_raw,
        nnunet_preprocessed=args.nnunet_preprocessed,
        nnunet_results=args.nnunet_results,
    )
