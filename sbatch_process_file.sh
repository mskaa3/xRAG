#!/bin/bash
#SBATCH --job-name=xRAG     # nazwa
#SBATCH --nodes=1                   # ilość węzłów
#SBATCH --cpus-per-gpu=4            # ilość cpu na zadanie
#SBATCH --time=4:00:00             # maksymalny czas wykonania zadania
#SBATCH --mem=256gb                 # ilość pamięci RAM
#SBATCH -p lem-gpu                     # partycja
#SBATCH --gres=gpu:hopper:4          # (ilość kart graficznych na węźle)
#SBATCH --verbose                   # wyświetlanie informacji o zadaniu
#SBATCH --exclude=r10-7

export NUM_PROCESSES=$(expr $SLURM_JOB_NUM_NODES \* 4)
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# REMOTE ARGS
# MODEL
# REMOTE_MODEL=s3min-tomasznaskret-1712063354/alignment/models/pllum-7b-v1-chat-v5
# # PROMPTS
REMOTE_DATA=s3min-tomasznaskret-1712063354/user/jmoska/xRAG_test_data
# # THIS IS SET AUTOMATICALLY BY THE SCRIPT
REMOTE_OUTPUT=s3min-tomasznaskret-1712063354/user/jmoska/xRAG_test_data/result



# RCLONE CONFIG NAME
export RCLONE_REMOTE_CONFIG=s3v2


# HUGGINGFACE
export HF_HOME=$TMPDIR/huggingface
export HF_TOKEN=$HF_TOKEN

# APPTAINER
export APPTAINER_TMPDIR=$TMPDIR/apptainer/
export APPTAINER_CACHEDIR=$TMPDIR/apptainer/
export APPTAINER_TRANSFORMERS_CACHE=$APPTAINER_TMPDIR

GENERATOR="python3 process_files.py run\
  --temperature 0.1 \
  --max_tokens 2048 \
  --data $REMOTE_DATA \
  --retriever_max_length 32000 \
  --xrag_token True \
  --output $REMOTE_OUTPUT \
  --file_name \"result\" \
  --save_format \"csv\" " \

# Execute in Apptainer (Singularity) container
APPTAINER_TMPDIR=/dev/shm/$SLURM_JOB_ID APPTAINER_CACHEDIR=/dev/shm/$SLURM_JOB_ID \
    apptainer exec --nv \
    --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
    ~/vllm_image.sif  \
    bash -c "$GENERATOR"




# APPTAINER_TMPDIR=/dev/shm/$SLURM_JOB_ID APPTAINER_CACHEDIR=/dev/shm/$SLURM_JOB_ID \
#     apptainer exec --nv \
#     --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
#     ~/vllm_image.sif  \
#     bash -c "$GENERATOR"
