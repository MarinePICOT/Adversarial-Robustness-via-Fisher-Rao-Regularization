
sbatch   --job-name=training \
  --gres=gpu:4 \
  --cpus-per-task=15 \
  --time=20:00:00 \
  --output=jobinfo/test_%j.out \
  --error=jobinfo/test_%j.err \
  --wrap="module purge; module load pytorch-gpu; python3 -m src.train --config config/mnist.yaml"
