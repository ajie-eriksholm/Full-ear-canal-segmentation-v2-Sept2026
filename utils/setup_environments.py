#!/usr/bin/env python3
"""
Environment Setup Script (Python version)
==========================================
This script automatically creates all required environments:
- seg_env      (venv)  - for P1 preprocessing
- landmark_env (venv)  - for P2-P4 preprocessing, inference, postprocessing
- metric_env   (conda) - for metric extraction (vmtk centerline analysis)

Usage:
    python setup_environments.py

Requirements:
    - Python 3.x installed
    - pip installed
    - conda installed (for metric_env / vmtk)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(text)
    print(f"{'='*60}{Colors.NC}\n")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.NC}")

def run_command(cmd, description=None):
    """Run a shell command and handle errors."""
    if description:
        print_info(description)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {cmd}")
        if e.stderr:
            print(e.stderr)
        return False

def create_venv(env_path, env_name):
    """Create a virtual environment."""
    if env_path.exists():
        print_warning(f"{env_name} already exists at: {env_path}")
        response = input("Do you want to remove and recreate it? (y/n): ").strip().lower()
        if response == 'y':
            print_info(f"Removing existing {env_name}...")
            shutil.rmtree(env_path)
        else:
            print_warning(f"Skipping {env_name} creation")
            return False
    
    print_info(f"Creating virtual environment at: {env_path}")
    if run_command(f"{sys.executable} -m venv {env_path}"):
        print_success(f"{env_name} created successfully")
        return True
    else:
        print_error(f"Failed to create {env_name}")
        sys.exit(1)

def install_requirements(env_path, req_file, env_name):
    """Install requirements in a virtual environment."""
    # Determine the activate script path
    if sys.platform == "win32":
        activate_script = env_path / "Scripts" / "activate.bat"
        pip_path = env_path / "Scripts" / "pip"
    else:
        activate_script = env_path / "bin" / "activate"
        pip_path = env_path / "bin" / "pip"
    
    print_info("Upgrading pip...")
    if not run_command(f"{pip_path} install --upgrade pip"):
        print_error("Failed to upgrade pip")
        return False
    
    print_info(f"Installing requirements from: {req_file}")
    print("This may take several minutes...")
    
    if run_command(f"{pip_path} install -r {req_file}"):
        print_success(f"{env_name} requirements installed successfully")
        return True
    else:
        print_error(f"Failed to install {env_name} requirements")
        return False

def main():
    """Main setup function."""
    # Get project root directory (parent of utils/)
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    
    # Environment paths (at project root)
    seg_env = project_root / "seg_env"
    landmark_env = project_root / "landmark_env"
    metric_env_name = "metric_env"
    metric_python_version = "3.11"
    
    # Requirement files (in env_req/ folder)
    seg_req = project_root / "env_req" / "seg_env_req.txt"
    landmark_req = project_root / "env_req" / "landmark_env_req.txt"
    metric_req = project_root / "env_req" / "metric_env_req.txt"
    
    # Print header
    print_header("ENVIRONMENT SETUP")
    
    print("This script will create the following environments:")
    print("  1. seg_env      (venv)  - for P1 preprocessing (TotalSegmentator, SimpleITK)")
    print("  2. landmark_env (venv)  - for P2-P4, inference, postprocessing (PyTorch, MONAI, PyVista)")
    print("  3. metric_env   (conda) - for metric extraction (vmtk, centerline analysis)")
    print()
    
    # Check Python version
    python_version = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_info(f"Found: {python_version}")
    print()
    
    # Check requirement files
    if not seg_req.exists():
        print_error(f"Requirement file not found: {seg_req}")
        sys.exit(1)
    
    if not landmark_req.exists():
        print_error(f"Requirement file not found: {landmark_req}")
        sys.exit(1)
    
    if not metric_req.exists():
        print_error(f"Requirement file not found: {metric_req}")
        sys.exit(1)
    
    print_success("Found all requirement files")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/n): ").strip().lower()
    if response != 'y':
        print_warning("Setup cancelled by user")
        sys.exit(0)
    
    # Create and setup seg_env
    print_header("STEP 1/6: Creating seg_env")
    skip_seg_env = not create_venv(seg_env, "seg_env")
    
    print_header("STEP 2/6: Installing seg_env requirements")
    if not skip_seg_env:
        if not install_requirements(seg_env, seg_req, "seg_env"):
            sys.exit(1)
    else:
        print_warning("Skipped seg_env installation")
    
    # Create and setup landmark_env
    print_header("STEP 3/6: Creating landmark_env")
    skip_landmark_env = not create_venv(landmark_env, "landmark_env")
    
    print_header("STEP 4/6: Installing landmark_env requirements")
    if not skip_landmark_env:
        if not install_requirements(landmark_env, landmark_req, "landmark_env"):
            sys.exit(1)
    else:
        print_warning("Skipped landmark_env installation")
    
    # Create and setup metric_env (conda)
    print_header("STEP 5/6: Creating metric_env (conda)")
    
    # Check conda is available
    if not shutil.which("conda"):
        print_error("conda not found. Please install Anaconda or Miniconda first.")
        print_info("https://docs.conda.io/en/latest/miniconda.html")
        sys.exit(1)
    
    # Check if metric_env already exists
    skip_metric_env = False
    result = subprocess.run("conda env list", shell=True, capture_output=True, text=True)
    if metric_env_name in result.stdout:
        print_warning(f"{metric_env_name} already exists")
        response = input("Do you want to remove and recreate it? (y/n): ").strip().lower()
        if response == 'y':
            print_info(f"Removing existing {metric_env_name}...")
            run_command(f"conda env remove -n {metric_env_name} -y")
        else:
            print_warning(f"Skipping {metric_env_name} creation")
            skip_metric_env = True
    
    if not skip_metric_env:
        print_info(f"Creating conda environment with Python {metric_python_version} and vmtk...")
        print("This may take several minutes as conda resolves dependencies...")
        if not run_command(
            f"conda create -n {metric_env_name} -c conda-forge --override-channels "
            f"python={metric_python_version} vmtk -y"
        ):
            print_error("Failed to create metric_env")
            sys.exit(1)
        print_success("metric_env created successfully")
    
    print_header("STEP 6/6: Installing metric_env pip requirements")
    
    if not skip_metric_env:
        # Install pip requirements and fix ITK symlinks via shell
        conda_hook = 'eval "$(conda shell.bash hook)"'
        install_cmd = (
            f'{conda_hook} && conda activate {metric_env_name} && '
            f'pip install -r {metric_req} && '
            f'cd "$CONDA_PREFIX/lib" && '
            f'for f in *-5.4.so.1; do link="${{f/-5.4.so.1/-5.3.so.1}}"; '
            f'[ ! -e "$link" ] && ln -s "$f" "$link"; done; '
            f'for f in *-5.4.so; do link="${{f/-5.4.so/-5.3.so}}"; '
            f'[ ! -e "$link" ] && ln -s "$f" "$link"; done; '
            f'cd "{project_root}"'
        )
        if run_command(install_cmd, "Installing pip packages and fixing ITK symlinks..."):
            print_success("metric_env requirements installed and ITK symlinks created")
        else:
            print_error("Failed to install metric_env requirements")
            sys.exit(1)
    else:
        print_warning("Skipped metric_env installation")
    
    # Print summary
    print_header("SETUP COMPLETE!")
    
    print("Virtual environments created:")
    print()
    
    if not skip_seg_env:
        print_success(f"seg_env:      {seg_env}")
        if sys.platform == "win32":
            print("  → Activate with: seg_env\\Scripts\\activate")
        else:
            print("  → Activate with: source seg_env/bin/activate")
        print("  → Used for: P1 preprocessing")
    else:
        print_warning("seg_env:      Skipped (already exists)")
    print()
    
    if not skip_landmark_env:
        print_success(f"landmark_env: {landmark_env}")
        if sys.platform == "win32":
            print("  → Activate with: landmark_env\\Scripts\\activate")
        else:
            print("  → Activate with: source landmark_env/bin/activate")
        print("  → Used for: P2-P4, inference, postprocessing")
    else:
        print_warning("landmark_env: Skipped (already exists)")
    print()
    
    if not skip_metric_env:
        print_success(f"metric_env:   (conda environment)")
        print("  → Activate with: conda activate metric_env")
        print("  → Used for: metric extraction (vmtk centerline analysis)")
    else:
        print_warning("metric_env:   Skipped (already exists)")
    print()
    
    print_info("You can now run the full pipeline using:")
    if sys.platform == "win32":
        print("  sh sh_files/run_full_pipeline.sh")
    else:
        print("  ./sh_files/run_full_pipeline.sh")
    print()
    
    print_header("READY TO GO!")

if __name__ == "__main__":
    main()
