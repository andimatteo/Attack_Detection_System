#!/usr/bin/env bash
#
# train.sh - Script interattivo per lanciare il training con DPVaDE/Transformer
# utilizzando Gum per una TUI piacevole.
#
# Usage: 
#   ./train.sh
#
# Requisiti:
#   - gum (https://github.com/charmbracelet/gum)
#   - python3
#   - i requisiti del tuo progetto (requirements.txt)

set -e

# 1) BENVENUTO
gum style --foreground 212 --border normal --padding "1 2" --margin "1" <<EOM
Benvenuto nello script di training DPVaDE + Mini-Transformer!
Qui potrai impostare i parametri di allenamento in modo interattivo.
EOM

# 2) NUMERO EPOCHS
gum style --bold "Seleziona il numero di epoche (default 10):"
epochs="$(gum input --placeholder "10")"
if [ -z "$epochs" ]; then
  epochs=10
fi

# 3) BATCH SIZE
gum style --bold "Seleziona la dimensione del batch (default 32):"
batch_size="$(gum input --placeholder "32")"
if [ -z "$batch_size" ]; then
  batch_size=32
fi

# 4) SCHEDULER
gum style --bold "Scegli un LR scheduler:"
scheduler=$(gum choose "none" "cosine" "onecycle")

# 5) DEVICE
gum style --bold "Su quale device vuoi allenare?"
device=$(gum choose "cuda" "cpu")

# 6) PATH CSV
gum style --bold "Inserisci il path al CSV da usare per il training (default: ../../data/dataset.csv):"
csv_path="$(gum input --placeholder "../../data/dataset.csv")"
if [ -z "$csv_path" ]; then
  csv_path="../../data/dataset.csv"
fi

# 7) DIRECTORY PER LOG E CHECKPOINT
gum style --bold "Directory dove salvare i log di TensorBoard? (default: ../../runs/transformer_dp_vade):"
log_dir="$(gum input --placeholder "../../runs/transformer_dp_vade")"
if [ -z "$log_dir" ]; then
  log_dir="../../runs/transformer_dp_vade"
fi

gum style --bold "Directory dove salvare i checkpoint? (default: ../../checkpoints):"
save_dir="$(gum input --placeholder "../../checkpoints")"
if [ -z "$save_dir" ]; then
  save_dir="../../checkpoints"
fi

# 8) RIEPILOGO
gum style --border normal --margin "1 0" --padding "1 2" --foreground 212 <<EOM
Parametri selezionati:

  - Epochs:       $epochs
  - Batch size:   $batch_size
  - Scheduler:    $scheduler
  - Device:       $device
  - CSV path:     $csv_path
  - Log dir:      $log_dir
  - Checkpoints:  $save_dir

Confermi di voler eseguire il training con questi parametri?
EOM

# 9) CONFERMA
if gum confirm --affirmative "Sì" --negative "No"; then
  gum spin --spinner dot --title "Avvio Training..." --hide-output -- \
  python3 ../src/transformer_dp_vade/train.py \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --scheduler "$scheduler" \
    --device "$device" \
    --csv_path "$csv_path" \
    --log_dir "$log_dir" \
    --save_dir "$save_dir"
  
  gum style --bold --foreground 46 "✅ Training completato!"
else
  gum style --bold --foreground 196 "❌ Training annullato!"
fi
