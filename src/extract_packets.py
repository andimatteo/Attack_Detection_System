import pyshark  # Libreria per analisi file pcapng
import pandas as pd  # Libreria per gestione dati tabulari
import os
import sys

# Funzione che estrae le caratteristiche desiderate da un singolo pacchetto
def extract_features(packet):
    try:
        features = {
            "tcp": int(hasattr(packet, 'tcp')),  # 1 se il pacchetto ha layer TCP
            "AckDat": int(packet.tcp.ack) if hasattr(packet, 'tcp') and hasattr(packet.tcp, 'ack') else 0,  # ACK Number
            "sHops": int(packet.ip.ttl) if hasattr(packet, 'ip') else 0,  # TTL del pacchetto IP
            "Seq": int(packet.tcp.seq) if hasattr(packet, 'tcp') and hasattr(packet.tcp, 'seq') else 0,  # Numero sequenza TCP
            "RST": int(packet.tcp.flags_reset) if hasattr(packet, 'tcp') and hasattr(packet.tcp, 'flags_reset') else 0,  # Flag reset TCP
            "TcpRtt": float(packet.tcp.analysis_ack_rtt) if hasattr(packet, 'tcp') and hasattr(packet.tcp, 'analysis_ack_rtt') else 0,  # TCP Round-Trip Time
            "REQ": int(hasattr(packet, 'http') and hasattr(packet.http, 'request_method')),  # Richiesta HTTP presente
            "dMeanPktSz": int(packet.length),  # Lunghezza pacchetto in bytes
            "Offset": int(packet.tcp.window_size) if hasattr(packet, 'tcp') else 0,  # TCP Window size
            "CON": int(packet.tcp.flags_syn) if hasattr(packet, 'tcp') else 0,  # TCP SYN flag
            "FIN": int(packet.tcp.flags_fin) if hasattr(packet, 'tcp') else 0,  # TCP FIN flag
            "sTtl": int(packet.ip.ttl) if hasattr(packet, 'ip') else 0,  # TTL sorgente
            "e": int(hasattr(packet, 'eth')),  # Ethernet layer presente
            "INT": int(hasattr(packet, 'tcp') and hasattr(packet.tcp, 'flags_push')),  # TCP PUSH flag
            "Mean": int(packet.length),  # Lunghezza pacchetto
            "Status": int(packet.http.response_code) if hasattr(packet, 'http') and hasattr(packet.http, 'response_code') else 0,  # HTTP Status
            "icmp": int(hasattr(packet, 'icmp')),  # ICMP presente
            "SrcTCPBase": int(packet.tcp.srcport) if hasattr(packet, 'tcp') else 0,  # Porta TCP sorgente
            "e_d": int(packet.eth.type, 16) if hasattr(packet, 'eth') else 0,  # Ethernet type
            "sMeanPktSz": int(packet.length),  # Dimensione media pacchetto sorgente
            "DstLoss": 0,  # Placeholder (necessita info esterna)
            "Loss": 0,  # Placeholder (necessita info esterna)
            "dTtl": int(packet.ip.ttl) if hasattr(packet, 'ip') else 0,  # TTL destinazione
            "SrcBytes": int(packet.captured_length),  # Bytes sorgente
            "TotBytes": int(packet.length)  # Bytes totali
        }
        return features
    except:
        return None  # Se errore nella lettura, ritorna None

# Processa file .pcapng e ritorna lista di caratteristiche
def process_pcapng(file_path, class_label):
    cap = pyshark.FileCapture(file_path, keep_packets=False)
    data = []
    
    for packet in cap:
        features = extract_features(packet)
        if features:
            features['class'] = class_label  # Aggiunge etichetta di classe
            data.append(features)
    
    cap.close()
    return data

# Funzione principale dello script
def main(input_folder, output_file):
    all_data = []
    
    # Itera su file nella cartella input
    for file in os.listdir(input_folder):
        if file.endswith('.pcapng'):
            class_label = file.split('_')[0]  # Imposta l'etichetta di classe dal nome file
            file_path = os.path.join(input_folder, file)
            print(f"Processando {file_path} (classe: {class_label})...")
            all_data.extend(process_pcapng(file_path, class_label))
    
    # Crea dataframe e salva su file CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"✅ Dati salvati correttamente in: {output_file}")

# Esecuzione script
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Utilizzo corretto: python3 extract_packets.py <cartella_input> <file_output.csv>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_file = sys.argv[2]
    
    main(input_folder, output_file)

