#!/usr/bin/env bash
set -euo pipefail

base_cmd="python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd_sw --model_s wrn_40_1 -a 0 -b 0.8 --trial 1 --sw_alpha 1"

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --model_s wrn_40_1 -a 0 -b 0.8 --trial 1

for tau in 0.4 0.6; do
  echo "Running sw_tau=${tau}..."
  ${base_cmd} --sw_tau "${tau}"
done