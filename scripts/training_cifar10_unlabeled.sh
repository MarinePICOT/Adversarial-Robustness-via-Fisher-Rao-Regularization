
sbatch   --account=scd@gpu\
   --job-name=training \
  --partition=gpu_p2 \
  --gres=gpu:4 \
  --no-requeue \
  --cpus-per-task=15 \
  --hint=nomultithread \
  --time=20:00 \
  --output=jobinfo/test_%j.out \
  --error=jobinfo/test_%j.err \
  --qos=qos_gpu-t3 \
  --wrap="module purge; module load pytorch-gpu; python -m src.train --config config/cifar10.yaml --opts unlabeled True aux_data_filename ti_500K_pseudo_labeled.pickle epochs 200 arch wrn-28-10"

#aux_data_filename ti_500K_pseudo_labeled.pickle
