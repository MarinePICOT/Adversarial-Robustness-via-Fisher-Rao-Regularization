
sbatch   --account=scd@gpu\
   --job-name=training \
  --partition=gpu_p2\
  --gres=gpu:4 \
  --no-requeue \
  --cpus-per-task=15 \
  --hint=nomultithread \
  --time=20:00 \
  --output=jobinfo/test_%j.out \
  --error=jobinfo/test_%j.err \
  --qos=qos_gpu-t3 \
  --wrap="module purge; module load pytorch-gpu; python3 -m src.train --config config/cifar100.yaml "

