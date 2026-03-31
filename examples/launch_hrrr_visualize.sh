#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J hrrr-vis
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -q debug
#SBATCH -o logs/hrrr-vis-%j.out
#SBATCH -e logs/hrrr-vis-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

module load PrgEnv-gnu
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

module unload darshan-runtime
module unload libfabric

source /lustre/orion/csc662/proj-shared/janet/miniconda3/etc/profile.d/conda.sh
conda activate orbit_main

module load libfabric/1.22.0
module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
module load SR_tools/devel-mpich8.1.31
module load aws-ofi-rccl/devel

echo $LD_LIBRARY_PATH

export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid

export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export TORCH_NCCL_HIGH_PRIORITY=1

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1
export MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_CONV_WINOGRAD=0

export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD/../src:$PYTHONPATH

export ORBIT_USE_DDSTORE=0
export LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6

# ============================================================
# HRRR Visualization
# ============================================================
# Change the checkpoint path and options as needed:
#   --index       : starting test sample index (default: 0)
#   --num_samples : how many samples to visualize (default: 5)
#   --save_dir    : output directory for images and numpy arrays

CHECKPOINT="/lustre/orion/csc662/proj-shared/janet/checkpoints/hrrr_refc_v2/checkpoint_epoch_0027.pt"
CONFIG="/ccs/home/janetw/diffusion/ORBIT-2-main/configs/hrrr_forecasting_v2.yaml"

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python /ccs/home/janetw/diffusion/ORBIT-2-main/examples/hrrr_visualize.py ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --index 0 \
    --num_samples 5 \
    --save_dir visualizations
