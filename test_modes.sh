#!/bin/bash
export PYTHONPATH="$(pwd):$(pwd)/app:${PYTHONPATH}"

echo "Testing all 3 model modes..."
echo ""

for mode in direct diffusion_fast diffusion_standard; do
  echo "Testing mode: $mode"
  python app/scripts/train.py \
    --model_mode $mode \
    --wavelet haar \
    --contr t2f \
    --data_dir ./datasets/BRATS2023/training \
    --batch_size 1 \
    --lr 1e-4 \
    --save_interval 999999 \
    --log_interval 1 \
    --lr_anneal_steps 5 2>&1 | grep -E "(MODE:|Error|Failed)" | head -5
  
  if [ $? -eq 0 ]; then
    echo "✅ $mode works"
  else
    echo "❌ $mode failed"
  fi
  echo ""
done
