#!/bin/bash
#SBATCH --job-name=stsb-finetuning
#SBATCH --partition=gpu_4_h100  # choose your partition (you can check which ones are available with sinfo_t_idle)
#SBATCH --time=00:30:00  	# computing time, after this, your job will be terminated
#SBATCH --nodes=1  		# compute nodes, for most jobs one is enough
#SBATCH --ntasks=1  		# number of tasks across all nodes
#SBATCH --gres=gpu:1		# claim 4 GPUs for your job (you should use more than one for e.g. model fine-tuning)	
#SBATCH --output="stsb_out"	# writes the outputs of your job to a file "mnli_out"
#SBATCH --mail-type=begin       # send email when job begins
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-type=fail        # send email when job fails
#SBATCH --mail-user=janjung@mail.uni-mannheim.de

module load devel/miniconda/4.9.2  # load miniconda

source ~/.bashrc

conda activate /pfs/work7/workspace/scratch/ma_pbhattar-test_teamproj/envs/fineTuning2  # activate your environment "fineTuning2"

# by default, the script uses multiple GPUs effectively if they are available

python run_glue.py --model_name_or_path roberta-base --task_name stsb --do_train --do_eval --max_seq_length 512 --per_device_eval_batch_size=8 --per_device_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir outputs/STS-B/model1

conda deactivate  # deactivate the conda environment


