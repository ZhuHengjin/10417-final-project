#!/usr/bin/env bash

# Run a CRD-SW student experiment and extract its TensorBoard logs to CSV.
set -eu
if set -o pipefail 2>/dev/null; then
  :
fi

TEACHER_PATH="./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth"
MODEL_S="resnet8x4"
MODEL_T="resnet32x4"
DATASET="cifar100"
DISTILL="crd_sw"
GAMMA="1"
ALPHA="0.0"
BETA="0.8"
TRIAL="1"
LOG_ROOT="logs"
CSV_ROOT="report/logs"

SW_ALPHA="${1:-1.0}"
SW_TAU="${2:-0.6}"

python3 train_student.py \
  --path_t "${TEACHER_PATH}" \
  --distill "${DISTILL}" \
  --model_s "${MODEL_S}" \
  -r "${GAMMA}" \
  -a "${ALPHA}" \
  -b "${BETA}" \
  --dataset "${DATASET}" \
  --trial "${TRIAL}" \
  --sw_alpha "${SW_ALPHA}" \
  --sw_tau "${SW_TAU}"

RUN_NAME="S:${MODEL_S}_T:${MODEL_T}_${DATASET}_${DISTILL}_r:${GAMMA}_a:${ALPHA}_b:${BETA}_sw:${SW_ALPHA}_tau:${SW_TAU}_${TRIAL}"
RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
CSV_OUT="${CSV_ROOT}/${RUN_NAME}.csv"

mkdir -p "${CSV_ROOT}"

python3 report/extract_log.py --run_dir "${RUN_DIR}" --output "${CSV_OUT}"

echo "Saved metrics CSV to ${CSV_OUT}"

python3 report/report_best_metrics.py --csv "${CSV_OUT}"
