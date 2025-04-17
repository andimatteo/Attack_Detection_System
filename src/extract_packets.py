#!/usr/bin/env python3
import pyshark
import pandas as pd
import os
import sys

# =============================================================================
# 1) FUNZIONI DI CONVERSIONE SICURA (per evitare errori "invalid literal for int() with base 10: 'False'")
# =============================================================================
def safe_int(value, default=0):
    """
    Converte 'value' in int in modo sicuro.
    - Se value è un bool, restituisce 1 (True) o 0 (False).
    - Se value è una stringa "True"/"False", restituisce 1 o 0.
    - Se value è una stringa numerica, prova int(value).
    - In caso di errore, restituisce 'default'.
    """
    if isinstance(value, bool):
        return 1 if value else 0
    
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return 1
        elif v == "false":
            return 0
        else:
            try:
                return int(v)
            except ValueError:
                return default

    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    """
    Converte 'value' in float in modo sicuro.
    - Se value è un bool, restituisce 1.0 o 0.0.
    - Se value è una stringa "True"/"False", restituisce 1.0 o 0.0.
    - Se value è una stringa numerica, prova float(value).
    - In caso di errore, restituisce 'default'.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return 1.0
        elif v == "false":
            return 0.0
        else:
            try:
                return float(v)
            except ValueError:
                return default

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# =============================================================================
# 2) MAPPATURA FILE -> CLASSI
# =============================================================================
class_mapping = {
    'Goldeneye': 'Goldeneye',
    'ICMPflood': 'ICMPflood',
    'SSH': 'normal',       # SSH benigno
    'SYNScan': 'SYNScan',
    'SYNflood': 'SYNflood',
    'Slowloris': 'Slowloris',
    'TCPConnect': 'TCPConnect',
    'Torshammer': 'Torshammer',
    'UDPScan': 'UDPScan',
    'UDPflood': 'UDPflood'
}


# =============================================================================
# 3) FUNZIONE PER ESTRARRE LE FEATURE DA UN SINGOLO PACCHETTO (SENZA IP)
#    Integra gestione di safe_int / safe_float
# =============================================================================
def extract_features(packet):
    try:
        features = {}

        # -----------------
        # Esempio di alcune feature come nei precedenti script
        # -----------------

        # a) Layer Ethernet
        has_eth = hasattr(packet, 'eth')
        features['e'] = int(has_eth)

        if has_eth and hasattr(packet.eth, 'type'):
            features['e_d'] = safe_int(packet.eth.type, 0)
        else:
            features['e_d'] = 0

        # b) Layer IP / TTL
        if hasattr(packet, 'ip') and hasattr(packet.ip, 'ttl'):
            features['sHops'] = safe_int(packet.ip.ttl)
        else:
            features['sHops'] = 0

        # Impostiamo sTtl e dTtl uguali a sHops (come spesso fatto per dataset flow-based)
        features['sTtl'] = features['sHops']
        features['dTtl'] = features['sHops']

        # c) TCP
        has_tcp = hasattr(packet, 'tcp')
        features['tcp'] = int(has_tcp)

        if has_tcp:
            # AckDat
            features['AckDat'] = safe_int(getattr(packet.tcp, 'ack', 0))
            # Seq
            features['Seq'] = safe_int(getattr(packet.tcp, 'seq', 0))
            # RST
            features['RST'] = safe_int(getattr(packet.tcp, 'flags_reset', 0))
            # TcpRtt
            features['TcpRtt'] = safe_float(getattr(packet.tcp, 'time_delta', 0.0))
            # Offset (TCP header length)
            features['Offset'] = safe_int(getattr(packet.tcp, 'hdr_len', 0))
            # CON (flags_syn)
            features['CON'] = safe_int(getattr(packet.tcp, 'flags_syn', 0))
            # FIN
            features['FIN'] = safe_int(getattr(packet.tcp, 'flags_fin', 0))
            # INT (push)
            features['INT'] = safe_int(getattr(packet.tcp, 'flags_push', 0))
            # SrcTCPBase (srcport)
            features['SrcTCPBase'] = safe_int(getattr(packet.tcp, 'srcport', 0))
        else:
            features['AckDat'] = 0
            features['Seq'] = 0
            features['RST'] = 0
            features['TcpRtt'] = 0.0
            features['Offset'] = 0
            features['CON'] = 0
            features['FIN'] = 0
            features['INT'] = 0
            features['SrcTCPBase'] = 0

        # d) HTTP
        features['REQ'] = 0
        features['Status'] = 0
        if hasattr(packet, 'http'):
            # REQ = 1 se c'è request_method
            if hasattr(packet.http, 'request_method'):
                features['REQ'] = 1
            # Status = response_code
            if hasattr(packet.http, 'response_code'):
                features['Status'] = safe_int(packet.http.response_code)

        # e) ICMP
        features['icmp'] = int(hasattr(packet, 'icmp'))

        # f) dMeanPktSz / Mean / sMeanPktSz (qui considerati uguali in base al singolo pacchetto)
        if hasattr(packet, 'length'):
            try:
                pkt_len = safe_int(packet.length, 0)
            except ValueError:
                pkt_len = 0
        else:
            pkt_len = 0

        features['dMeanPktSz'] = pkt_len
        features['Mean'] = pkt_len
        features['sMeanPktSz'] = pkt_len

        # g) DstLoss / Loss (placeholder = 0)
        features['DstLoss'] = 0
        features['Loss'] = 0

        # h) SrcBytes / TotBytes
        if hasattr(packet, 'captured_length'):
            features['SrcBytes'] = safe_int(packet.captured_length, 0)
        else:
            features['SrcBytes'] = 0
        features['TotBytes'] = pkt_len

        return features

    except Exception as e:
        print(f"Errore nell'estrazione delle caratteristiche: {e}")
        return None


# =============================================================================
# 4) FUNZIONE PER PROCESSARE FILE PCAPNG E SALVARE SU CSV
# =============================================================================
def process_pcapng(file_path, class_label, output_file):
    cap = pyshark.FileCapture(file_path, keep_packets=False)
    data = []
    packet_count = 0

    print(f"\tInizio elaborazione pacchetti per {file_path}")
    for packet in cap:
        features = extract_features(packet)
        if features:
            features['class'] = class_label
            data.append(features)

        packet_count += 1
        # Scrittura ogni 1000 pacchetti
        if packet_count % 1000 == 0:
            print(f"\t\t{packet_count} pacchetti elaborati...")
            pd.DataFrame(data).to_csv(output_file, mode='a',
                                      header=not os.path.exists(output_file),
                                      index=False)
            data.clear()

    # Scrittura finale
    if data:
        pd.DataFrame(data).to_csv(output_file, mode='a',
                                  header=not os.path.exists(output_file),
                                  index=False)

    cap.close()
    print(f"\tCompletato {file_path}: {packet_count} pacchetti elaborati")


# =============================================================================
# 5) MAIN
# =============================================================================
def main(input_folder, output_file):
    # Se esiste già, lo rimuoviamo
    if os.path.exists(output_file):
        os.remove(output_file)

    for file in os.listdir(input_folder):
        if file.endswith('.pcapng'):
            file_prefix = file.split('_')[0]
            class_label = class_mapping.get(file_prefix, 'unknown')
            file_path = os.path.join(input_folder, file)

            print(f"✅ Inizio elaborazione file: {file_path} (classe: {class_label})")
            process_pcapng(file_path, class_label, output_file)
            print(f"✅ Elaborazione completata per {file_path}")

    print(f"🎉 Dati salvati correttamente in: {output_file}")


# =============================================================================
# 6) ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Utilizzo corretto: python3 extract_packets.py <cartella_input> <file_output.csv>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_file = sys.argv[2]

    main(input_folder, output_file)
