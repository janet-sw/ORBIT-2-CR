#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J hrrr-diff-vis
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -o logs/hrrr-diff-vis-%j.out
#SBATCH -e logs/hrrr-diff-vis-%j.out

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
# HRRR Diffusion Visualization
# ============================================================
# Options:
#   --diff_checkpoint : path to trained diffusion model checkpoint
#   --num_steps       : Heun sampling steps (20=fast, 30=good, 50=best)
#   --num_samples     : how many test samples to visualize
#   --index           : starting test sample index
#   --save_dir        : output directory
#   --warm_start      : use SDEdit warm-start (recommended)
#   --sigma_start     : starting sigma for warm-start (default: 2.0)

CONFIG="/ccs/home/janetw/diffusion/ORBIT-2-CR/configs/hrrr_diffusion_refine.yaml"
DIFF_CKPT="/lustre/orion/csc662/proj-shared/janet/checkpoints/hrrr_refc_diffusion_v2/checkpoint_epoch_0034.pt"

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python /ccs/home/janetw/diffusion/ORBIT-2-CR/examples/hrrr_diffusion_visualize.py ${CONFIG} \
    --diff_checkpoint ${DIFF_CKPT} \
    --num_steps 30 \
    --warm_start --sigma_start 2.0 \
    --index 0 \
    --num_samples 5 \
    --save_dir visualizations_diffusion