#!/bin/sh
#SBATCH --job-name=chris_test
#SBATCH --partition=a100        # sinfo
#SBATCH --account=master        # sacctmgr show user $USER accounts
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/out/%A.out
#SBATCH --mail-user=christopher.vibe@hiof.no
#SBATCH --mail-type=ALL

: '
    tips and tricks from ntnu walkthrough:
    https://www.hpc.ntnu.no/idun/hardware/

    module avail # list the modules
    module spider CUDA # check available CUDA modules

    module spider Docker  # Check if Docker module exists
    module spider Singularity  # Check if Singularity module exists
'

WORKDIR=${SLURM_SUBMIT_DIR}

init_environment () {
    cd ${WORKDIR}
    echo "We are running from this directory: $SLURM_SUBMIT_DIR"
    echo "The name of the job is: $SLURM_JOB_NAME"
    echo "The job ID is $SLURM_JOB_ID"
    echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
    echo "Number of nodes: $SLURM_JOB_NUM_NODES"
    echo "We are using $SLURM_CPUS_ON_NODE cores"
    echo "We are using $SLURM_CPUS_ON_NODE cores per node"
    echo "Total of $SLURM_NTASKS cores"
}

init_environment | tee out/install_log.txt 

# Assuming Singularity is installed on the HPC cluster
singularity run --nv docker://your_docker_image python test.py | tee out/run_log.txt
uname -a
