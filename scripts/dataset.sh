#!/bin/bash

# Controlla se gum è installato
if ! command -v gum &> /dev/null; then
    echo "❌ gum non trovato! Consulta il README per l'installazione."
    exit 1
fi

RAW_DIR="../data/raw"
PROCESSED_DIR="../data/processed"
SRC_DIR="../src"

clear

# Mostra i bottoni grandi side-by-side
gum join --horizontal --align center \
"$(gum style \
	--foreground 212 --border-foreground 212 --border double \
	--align center --width 30 --margin "1 2" --padding "2 4" \
	'📥 DOWNLOAD')" \
"$(gum style \
	--foreground 212 --border-foreground 212 --border double \
	--align center --width 30 --margin "1 2" --padding "2 4" \
	'⚙️ PREPROCESS')"

# Fai scegliere il bottone all'utente
ACTION=$(gum choose "📥 DOWNLOAD" "⚙️ PREPROCESS")

if [ "$ACTION" = "📥 DOWNLOAD" ]; then
    mkdir -p "$RAW_DIR"

    gum spin --spinner dot --title "Scaricando BS1_GTP_removed..." -- wget --content-disposition "https://download.fairdata.fi:443/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NDUxNjg1OTYsImRhdGFzZXQiOiI5ZDEzZWYyOC0yY2E3LTQ0YjAtOTk1MC0yMjUzNTlhZmFjNjUiLCJmaWxlIjoiL0JTMV9HVFBfcmVtb3ZlZC56aXAiLCJwcm9qZWN0IjoiMjAwNjkzOCIsInJhbmRvbV9zYWx0IjoiNjJhNzU3NjcifQ.doI7fqfRg-jpBlB35Wbd0xD-DLA15FZSU-OV_WaKZTc" -P "$RAW_DIR"

    gum spin --spinner dot --title "Scaricando BS2_GTP_removed..." -- wget --content-disposition "https://download.fairdata.fi:443/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NDUxNjg2MTgsImRhdGFzZXQiOiI5ZDEzZWYyOC0yY2E3LTQ0YjAtOTk1MC0yMjUzNTlhZmFjNjUiLCJmaWxlIjoiL0JTMiBHVFAgcmVtb3ZlZC56aXAiLCJwcm9qZWN0IjoiMjAwNjkzOCIsInJhbmRvbV9zYWx0IjoiNzdiMmU3NjAifQ.z_GiMbZaNwgHlhdwD2MGQSAsCitiVe5bwi862zgzEZI" -P "$RAW_DIR"

    gum spin --spinner dot --title "Estrazione ZIP..." -- unzip -o "$RAW_DIR/*.zip" -d "$RAW_DIR"

    gum style --foreground 46 "✅ Download completato!"

elif [ "$ACTION" = "⚙️ PREPROCESS" ]; then
    mkdir -p "$PROCESSED_DIR"

    INPUT_DIR=$(gum input --placeholder "Cartella input dei file pcapng" --value "$RAW_DIR")
    OUTPUT_FILE=$(gum input --placeholder "Nome file CSV output" --value "$PROCESSED_DIR/packets_features.csv")

    gum spin --spinner line --title "⚙️ Elaborazione..." -- python3 "$SRC_DIR/extract_packets.py" "$INPUT_DIR" "$OUTPUT_FILE"

    gum style --foreground 46 "✅ Dataset elaborato in: $OUTPUT_FILE"
fi

