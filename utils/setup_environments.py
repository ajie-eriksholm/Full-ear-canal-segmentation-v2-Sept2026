#!/usr/bin/env python3
"""
Environment Setup Script (Python version)
==========================================
This script automatically creates both required virtual environments:
- seg_env (for P1 preprocessing)
- landmark_env (for P2, P3, P4 preprocessing)

Usage:
    python setup_environments.py

Requirements:
    - Python 3.x installed
    - pip installed
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
    
    # Requirement files (in env_req/ folder)
    seg_req = project_root / "env_req" / "seg_env_req.txt"
    landmark_req = project_root / "env_req" / "landmark_env_req.txt"
    
    # Print header
    print_header("PREPROCESSING ENVIRONMENT SETUP")
    
    print("This script will create two virtual environments:")
    print("  1. seg_env      - for P1 preprocessing (TotalSegmentator, SimpleITK)")
    print("  2. landmark_env - for P2-P4 preprocessing (PyTorch, MONAI, PyVista)")
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
    
    print_success("Found both requirement files")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/n): ").strip().lower()
    if response != 'y':
        print_warning("Setup cancelled by user")
        sys.exit(0)
    
    # Create and setup seg_env
    print_header("STEP 1/4: Creating seg_env")
    skip_seg_env = not create_venv(seg_env, "seg_env")
    
    print_header("STEP 2/4: Installing seg_env requirements")
    if not skip_seg_env:
        if not install_requirements(seg_env, seg_req, "seg_env"):
            sys.exit(1)
    else:
        print_warning("Skipped seg_env installation")
    
    # Create and setup landmark_env
    print_header("STEP 3/4: Creating landmark_env")
    skip_landmark_env = not create_venv(landmark_env, "landmark_env")
    
    print_header("STEP 4/4: Installing landmark_env requirements")
    if not skip_landmark_env:
        if not install_requirements(landmark_env, landmark_req, "landmark_env"):
            sys.exit(1)
    else:
        print_warning("Skipped landmark_env installation")
    
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
        print("  → Used for: P2, P3, P4 preprocessing")
    else:
        print_warning("landmark_env: Skipped (already exists)")
    print()
    
    print_info("You can now run the preprocessing pipeline using:")
    if sys.platform == "win32":
        print("  sh sh_files/run_full_preprocessing.sh")
    else:
        print("  ./sh_files/run_full_preprocessing.sh")
    print()
    
    print_header("READY TO PREPROCESS!")

if __name__ == "__main__":
    main()
