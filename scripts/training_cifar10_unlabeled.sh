
sbatch   --job-name=training \
  --gres=gpu:4 \
  --cpus-per-task=15 \
  --time=20:00:00 \
  --output=jobinfo/test_%j.out \
  --error=jobinfo/test_%j.err \
  --wrap="module purge; module load pytorch-gpu; python -m src.train --config config/cifar10.yaml --opts unlabeled True aux_data_filename ti_500K_pseudo_labeled.pickle epochs 200 arch wrn-28-10"

#aux_data_filename ti_500K_pseudo_labeled.pickle
