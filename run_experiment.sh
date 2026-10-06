#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="${1:-}"
STAGE="${2:-all}"
PROFILE="${3:-full}"

if [[ "$BENCHMARK" != "molecule" && "$BENCHMARK" != "polymer" ]]; then
  echo "Usage: $0 {molecule|polymer} {prepare|pt|sft|sdc|all} [full|minimal]" >&2
  exit 2
fi
if [[ ! "$STAGE" =~ ^(prepare|pt|sft|sdc|all)$ ]]; then
  echo "Unknown stage: $STAGE" >&2
  exit 2
fi
if [[ "$PROFILE" != "full" && "$PROFILE" != "minimal" ]]; then
  echo "Profile must be full or minimal." >&2
  exit 2
fi

export GRAPHFDM_ROOT="$ROOT"
export GRAPHFDM_DATA_ROOT="${GRAPHFDM_DATA_ROOT:-$ROOT/data}"
export GRAPHFDM_OUTPUT_ROOT="${GRAPHFDM_OUTPUT_ROOT:-$ROOT/outputs}"
mkdir -p "$GRAPHFDM_OUTPUT_ROOT"
cd "$ROOT"

latest_checkpoint() {
  local directory="$1"
  local selected
  selected="$(find "$directory" -type f -name '*.ckpt' ! -name 'last.ckpt' 2>/dev/null | sort | tail -n 1)"
  if [[ -z "$selected" ]]; then
    selected="$(find "$directory" -type f -name '*.ckpt' 2>/dev/null | sort | tail -n 1)"
  fi
  if [[ -z "$selected" ]]; then
    echo "No checkpoint found under $directory" >&2
    exit 1
  fi
  python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$selected"
}

prepare_data() {
  if [[ "$BENCHMARK" == "molecule" ]]; then
    python scripts/prepare_molecule.py --profile "$PROFILE" --data_root "$GRAPHFDM_DATA_ROOT"
  else
    python scripts/prepare_polymer.py \
      --data_root "$GRAPHFDM_DATA_ROOT" \
      --tokenizer_dir "$ROOT/assets/tokenizers/polymer"
  fi
}

train_pt() {
  local overrides=()
  if [[ "$PROFILE" == "minimal" ]]; then
    overrides=(
      train.n_epochs=1 train.batch_size=8 train.check_val_every_n_epoch=1
      model.hidden_size=128 model.depth=2 model.num_heads=4 model.diffusion_steps=10
      +general.limit_train_batches=2 +general.limit_val_batches=1
    )
  fi
  python main_pt.py --config-name "pt_${BENCHMARK}" "${overrides[@]}"
}

train_sft() {
  export GRAPHFDM_PT_CHECKPOINT="${GRAPHFDM_PT_CHECKPOINT:-$(latest_checkpoint "$GRAPHFDM_OUTPUT_ROOT/$BENCHMARK/pt/checkpoints")}" 
  local overrides=()
  if [[ "$PROFILE" == "minimal" ]]; then
    overrides=(
      train.n_epochs=1 train.batch_size=8 train.check_val_every_n_epoch=1
      model.hidden_size=128 model.depth=2 model.num_heads=4 model.diffusion_steps=10
      general.validation_sampling=false general.checkpoint_monitor=null
      +general.limit_train_batches=2 +general.limit_val_batches=1
    )
  fi
  python main_pt.py --config-name "sft_${BENCHMARK}" "${overrides[@]}"
}

train_graphfdm() {
  export GRAPHFDM_SFT_CHECKPOINT="${GRAPHFDM_SFT_CHECKPOINT:-$(latest_checkpoint "$GRAPHFDM_OUTPUT_ROOT/$BENCHMARK/sft/checkpoints")}" 
  export GRAPHFDM_SFT_CONFIG="${GRAPHFDM_SFT_CONFIG:-$GRAPHFDM_OUTPUT_ROOT/$BENCHMARK/sft/.hydra/config.yaml}"
  local overrides=()
  if [[ "$PROFILE" == "minimal" ]]; then
    overrides=(
      online.num_rounds=1 online.k=16
      trust_region.marginal_kl=0.005 trust_region.conditional_kl=0.005
      collection.batch_size=64 collection.oracle_batch_size=256
      graph_train.epochs_per_round=1 graph_train.batch_size=64
      controller_train.epochs_per_round=1 controller_train.batch_size=64
      eval.val_total_samples=16 eval.test_total_samples=16 eval.batch_size=16
      eval.oracle_batch_size=256 eval.compute_distribution_metrics=false
      sampling.diffusion_steps=10
    )
  fi
  python graphfdm/train.py --config "configs/graphfdm_${BENCHMARK}.yaml" "${overrides[@]}"
}

case "$STAGE" in
  prepare) prepare_data ;;
  pt) train_pt ;;
  sft) train_sft ;;
  sdc) train_graphfdm ;;
  all)
    prepare_data
    train_pt
    train_sft
    train_graphfdm
    ;;
esac
