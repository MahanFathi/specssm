#!/bin/bash

# SLURM SUBMISSION SCRIPT TEMPLATE

# Job Name:
##SBATCH --job-name=my_job_name

# Partition:
##SBATCH --partition=main  # Replace with your desired partition

# Time required in days-hours:minutes:seconds
##SBATCH --time=1-00:00:00  # 1 day, 0 hours, 0 minutes, 0 seconds

# Memory required per node:
##SBATCH --mem=4G  # Adjust as needed

# Number of nodes:
##SBATCH --nodes=1  # Adjust based on your requirements

# Number of tasks to be launched per Node:
##SBATCH --ntasks-per-node=1  # Adjust as needed

# Set output and error files
##SBATCH --output=my_job_output_%j.txt  # %j will be replaced with the job ID
##SBATCH --error=my_job_error_%j.txt

# Send email at start, end, and abortion of execution:
##SBATCH --mail-type=ALL  # Options are BEGIN, END, FAIL, ALL

# Email address to receive notifications (change to your email):
##SBATCH --mail-user=your_email@example.com

# Specify a job array if needed (uncomment to enable):
##SBATCH --array=0-3  # Creates a job array with 4 tasks

# Specify any constraints (uncomment to enable):
##SBATCH --constraint=[constraint_name]

# Specify account for resource allocation (uncomment to enable, if required):
##SBATCH --account=[your_account_name]

# Set a quality of service, if needed (uncomment to enable):
##SBATCH --qos=[quality_of_service]

# --------------------------------------------------------------------------------

#SBATCH --job-name=specssm
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --output=log.txt

#SBATCH --mail-user=mahan.fathi@mila.quebec
#SBATCH --mail-type=ALL  # Options are BEGIN, END, FAIL, ALL
#SBATCH --mail-user=mahanedx@gmail.com

# Load any modules or software if needed
module load libffi
module load singularity

# copy over the container to compute node
rsync -avz $SCRATCH/specssm_v1.0.sif $SLURM_TMPDIR

# copty the code over
rsync -avz . $SLURM_TMPDIR

# extract the dataset, etc.
rm -rf $SLURM_TMPDIR/raw_datasets && mkdir $SLURM_TMPDIR/raw_datasets
# NOTE: The command below is a bit slow but the good news is that,
# typically, the `cache_dir` already exists in . and will get copied over to $SLURM_TMPDIR.
# tar -xf $SCRATCH/lra_release.gz -C $SLURM_TMPDIR/raw_datasets

# Generate a timestamp using the `date` command
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Set the base dir for the log files
LOG_BASE="$SCRATCH/logs"

# Define the full path for the output and error files
OUTPUT_FILE="${LOG_BASE}/${TIMESTAMP}.txt"

# Execute your script
EXPERIMENT_NAME=$1 # from {aan, cifar, imdb, listops, pathfinder, pathx}

# Construct the script path
SCRIPT_PATH="${SLURM_TMPDIR}/bin/${EXPERIMENT_NAME}.sh"

#singularity exec --nv --bind $SLURM_TMPDIR/raw_datasets:/home/raw_datasets -H $HOME/codeporn/specssm:/home $SLURM_TMPDIR/specssm_v1.0.sif bash "$SCRIPT_PATH"
srun --output=$OUTPUT_FILE singularity exec --nv --env WANDB_API_KEY=bc83aef47b057ca6f98927f1917c515a7d4bae61 --home $SLURM_TMPDIR $SLURM_TMPDIR/specssm_v1.0.sif bash "$SCRIPT_PATH"

# thats my wandb api key knock yourself out

# sample command: sbatch claunch.sh cifar
