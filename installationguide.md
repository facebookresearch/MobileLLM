# Project Setup Guide

This guide will help you set up the environment required to run the pretraining script. It covers installation on both Windows (using WSL) and Linux.

## Prerequisites

- **WSL (Windows Subsystem for Linux)**: Ensure you have WSL installed if you are using Windows.
- **Linx or Mac OS**.

### For Windows Users (Using WSL)

1. **Install WSL**

   Open a Command Prompt or PowerShell as Administrator and run:
   ```sh
   wsl --install
   #Navigate to your folder in window drive and run the command

2. **Install packages**

   Get into your WSL 
   ```sh
   #Access your window directory where your requirements file is avaliable. 
   0- sudo apt-get update
   1- pip install -r requirements.txt
   2- pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 #Make sure you have same version install on your comptuer from nvidia website
   #Test Installation using
   4- python -c "import torch; print(torch.cuda.nccl.version())"
   5- python -c "import torch; print(torch.cuda.is_available())"

3. **Run the code**
    ```sh
    ./pretrain.sh
  
 









