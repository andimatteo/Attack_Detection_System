#!/usr/bin/env python3

import pyshark
import pandas as pd
import os
import sys
import math

# =============================================================================
# 1) FUNZIONI DI CONVERSIONE SICURA (per evitare errori “invalid literal for int() with base 10: 'False'”)
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
# 2) FUNZIONI UTILI PER CALCOLARE ENTROPIA E ZERO-RATIO DAL PAYLOAD
# =============================================================================
def hex_string_to_bytes(hex_str):
    """
    Converte una stringa esadecimale (es. 'abcd1234') in bytes grezzi (b'...').
    Ignora eventuali separatori tipo ':' se presenti.
    """
    # Rimuove caratteri non validi, come ':'
    filtered = "".join(c for c in hex_str if c in "0123456789abcdefABCDEF")
    if len(filtered) % 2 != 0:
        # Se lunghezza dispari, tronca un nibble (caso raro)
        filtered = filtered[:-1]
    try:
        return bytes.fromhex(filtered)
    except ValueError:
        return b''

def payload_entropy(hex_payload):
    """
    Calcola l'entropia di Shannon del payload a partire da una stringa esadecimale.
    Se non valido o vuoto, restituisce 0.0.
    """
    raw = hex_string_to_bytes(hex_payload)
    if len(raw) == 0:
        return 0.0

    # Conta la frequenza di ogni byte [0..255]
    freq = [0]*256
    for b in raw:
        freq[b] += 1

    ent = 0.0
    length = float(len(raw))
    for f in freq:
        if f > 0:
            p = f/length
            ent -= p * math.log2(p)
    return ent

def payload_zero_ratio(hex_payload):
    """
    Ritorna la percentuale (0..1) di byte a zero rispetto alla lunghezza totale del payload.
    """
    raw = hex_string_to_bytes(hex_payload)
    if len(raw) == 0:
        return 0.0
    zero_count = raw.count(0)
    return zero_count / float(len(raw))


# =============================================================================
# 3) MAPPATURA FILE -> CLASSI (esempio)
# =============================================================================
class_mapping = {
    'Goldeneye': 'Goldeneye',
    'ICMPflood': 'ICMPflood',
    'SSH': 'normal',
    'SYNScan': 'SYNScan',
    'SYNflood': 'SYNflood',
    'Slowloris': 'Slowloris',
    'TCPConnect': 'TCPConnect',
    'Torshammer': 'Torshammer',
    'UDPScan': 'UDPScan',
    'UDPflood': 'UDPflood'
}


# =============================================================================
# 4) FUNZIONE PER ESTRARRE 25 FEATURE DA UN SINGOLO PACCHETTO (SENZA IP)
# =============================================================================
def extract_features(packet):
    """
    Estrae 25 caratteristiche “allo stato dell’arte” da un singolo pacchetto:
    1) proto_number         (es. 6 = TCP, 17 = UDP, 1 = ICMP, 0 = altrimenti)
    2) total_length         (packet.length)
    3) captured_length      (packet.captured_length)
    4) ip_ttl               (se IP)
    5) ip_flags             (in base 16 o int)
    6) ip_frag_offset       (se disponibile)
    7) tcp_flags            (bitmask combinata su 6 bit: URG, ACK, PSH, RST, SYN, FIN)
    8) ack_flag             (1/0)
    9) syn_flag             (1/0)
    10) fin_flag            (1/0)
    11) rst_flag            (1/0)
    12) psh_flag            (1/0)
    13) urg_flag            (1/0)
    14) ack_number          (safe_int)
    15) seq_number          (safe_int)
    16) window_size         (safe_int)
    17) src_port            (TCP/UDP)
    18) dst_port            (TCP/UDP)
    19) icmp_type           (se ICMP)
    20) icmp_code           (se ICMP)
    21) transport_payload_length (tcp.len o udp.length, altrimenti 0)
    22) time_delta          (frame o tcp time_delta)
    23) payload_entropy     (entropia del payload L4 in esadecimale)
    24) payload_zero_ratio  (% byte = 0)
    25) proto_count         (numero protocolli nel frame_info, de-duplicati)

    => + la colonna 'class'
    """
    try:
        features = {}
        
        # 1) proto_number
        #    Proviamo a prendere ip.proto (se c'è), altrimenti 0
        proto_num = 0
        if hasattr(packet, 'ip') and hasattr(packet.ip, 'proto'):
            proto_num = safe_int(packet.ip.proto, 0)
        features['proto_number'] = proto_num

        # 2) total_length
        pkt_len = safe_int(getattr(packet, 'length', 0))
        features['total_length'] = pkt_len

        # 3) captured_length
        cap_len = safe_int(getattr(packet, 'captured_length', 0))
        features['captured_length'] = cap_len

        # 4) ip_ttl
        ip_ttl = 0
        if hasattr(packet, 'ip') and hasattr(packet.ip, 'ttl'):
            ip_ttl = safe_int(packet.ip.ttl, 0)
        features['ip_ttl'] = ip_ttl

        # 5) ip_flags
        ip_flags = 0
        if hasattr(packet, 'ip') and hasattr(packet.ip, 'flags'):
            # ip.flags può essere '0xXXXX' -> safe_int su base 16
            # Oppure direttamente un numero -> safe_int standard
            val = packet.ip.flags
            if val.startswith('0x'):
                try:
                    ip_flags = int(val, 16)
                except ValueError:
                    ip_flags = 0
            else:
                ip_flags = safe_int(val, 0)
        features['ip_flags'] = ip_flags

        # 6) ip_frag_offset
        ip_frag_off = 0
        if hasattr(packet, 'ip') and hasattr(packet.ip, 'frag_offset'):
            ip_frag_off = safe_int(packet.ip.frag_offset, 0)
        features['ip_frag_offset'] = ip_frag_off

        # 7) tcp_flags (bitmask): URG(32), ACK(16), PSH(8), RST(4), SYN(2), FIN(1)
        tcp_flags_bitmask = 0
        # 8..13) singoli flag
        ack_flag, syn_flag, fin_flag, rst_flag, psh_flag, urg_flag = 0,0,0,0,0,0

        # 14) ack_number
        ack_number = 0
        # 15) seq_number
        seq_number = 0
        # 16) window_size
        window_size = 0

        # 17) src_port
        src_port = 0
        # 18) dst_port
        dst_port = 0

        # 19) icmp_type
        icmp_type = 0
        # 20) icmp_code
        icmp_code = 0

        # 21) transport_payload_length (tcp.len o udp.length)
        transport_payload_len = 0

        # 23) payload_entropy
        # 24) payload_zero_ratio
        pay_entropy = 0.0
        pay_zero_ratio = 0.0

        # Verifica protocolli
        if hasattr(packet, 'tcp'):
            # -------- TCP ----------
            # ack_flag
            ack_flag = safe_int(getattr(packet.tcp, 'flags_ack', 0))
            # syn_flag
            syn_flag = safe_int(getattr(packet.tcp, 'flags_syn', 0))
            # fin_flag
            fin_flag = safe_int(getattr(packet.tcp, 'flags_fin', 0))
            # rst_flag
            rst_flag = safe_int(getattr(packet.tcp, 'flags_reset', 0))
            # psh_flag
            psh_flag = safe_int(getattr(packet.tcp, 'flags_push', 0))
            # urg_flag
            urg_flag = safe_int(getattr(packet.tcp, 'flags_urg', 0))

            # Costruiamo la bitmask
            # URG(32), ACK(16), PSH(8), RST(4), SYN(2), FIN(1)
            tcp_flags_bitmask = (
                (urg_flag << 5) |
                (ack_flag << 4) |
                (psh_flag << 3) |
                (rst_flag << 2) |
                (syn_flag << 1) |
                (fin_flag)
            )

            # ack_number / seq_number
            ack_number = safe_int(getattr(packet.tcp, 'ack', 0))
            seq_number = safe_int(getattr(packet.tcp, 'seq', 0))

            # window_size
            window_size = safe_int(getattr(packet.tcp, 'window_size_value', 0))

            # porte
            src_port = safe_int(getattr(packet.tcp, 'srcport', 0))
            dst_port = safe_int(getattr(packet.tcp, 'dstport', 0))

            # lunghezza payload tcp
            transport_payload_len = safe_int(getattr(packet.tcp, 'len', 0))

            # payload in esadecimale => packet.tcp.payload
            hex_payload = getattr(packet.tcp, 'payload', None)
            if hex_payload:
                pay_entropy = payload_entropy(hex_payload)
                pay_zero_ratio = payload_zero_ratio(hex_payload)

        elif hasattr(packet, 'udp'):
            # -------- UDP ----------
            src_port = safe_int(getattr(packet.udp, 'srcport', 0))
            dst_port = safe_int(getattr(packet.udp, 'dstport', 0))
            # In molti .pcap, udp.length è la dimensione del payload + header UDP (8 byte)
            transport_payload_len = safe_int(getattr(packet.udp, 'length', 0))

            # payload
            hex_payload = getattr(packet.udp, 'payload', None)
            if hex_payload:
                pay_entropy = payload_entropy(hex_payload)
                pay_zero_ratio = payload_zero_ratio(hex_payload)

        elif hasattr(packet, 'icmp'):
            # -------- ICMP ----------
            icmp_type = safe_int(getattr(packet.icmp, 'type', 0))
            icmp_code = safe_int(getattr(packet.icmp, 'code', 0))
            # In alcuni casi c'è anche icmp.ident, icmp.seq
            # Non sempre c'è un field length. Lo settiamo a 0
            # Se vuoi, puoi provare `getattr(packet.icmp, 'payload', None)`

            hex_payload = getattr(packet.icmp, 'payload', None)
            if hex_payload:
                pay_entropy = payload_entropy(hex_payload)
                pay_zero_ratio = payload_zero_ratio(hex_payload)

        # 22) time_delta (usiamo frame_info se tcp.time_delta non c’è)
        time_delta = 0.0
        if hasattr(packet, 'tcp') and hasattr(packet.tcp, 'time_delta'):
            time_delta = safe_float(packet.tcp.time_delta, 0.0)
        else:
            if hasattr(packet, 'frame_info') and hasattr(packet.frame_info, 'time_delta'):
                time_delta = safe_float(packet.frame_info.time_delta, 0.0)

        # 25) proto_count = numero protocolli nel frame_info
        proto_count = 0
        if hasattr(packet, 'frame_info') and hasattr(packet.frame_info, 'protocols'):
            splitted = set(packet.frame_info.protocols.split(':'))
            proto_count = len(splitted)

        # Ora salviamo tutto nel dizionario
        features['proto_number'] = proto_num
        features['total_length'] = pkt_len
        features['captured_length'] = cap_len
        features['ip_ttl'] = ip_ttl
        features['ip_flags'] = ip_flags
        features['ip_frag_offset'] = ip_frag_off
        features['tcp_flags'] = tcp_flags_bitmask
        features['ack_flag'] = ack_flag
        features['syn_flag'] = syn_flag
        features['fin_flag'] = fin_flag
        features['rst_flag'] = rst_flag
        features['psh_flag'] = psh_flag
        features['urg_flag'] = urg_flag
        features['ack_number'] = ack_number
        features['seq_number'] = seq_number
        features['window_size'] = window_size
        features['src_port'] = src_port
        features['dst_port'] = dst_port
        features['icmp_type'] = icmp_type
        features['icmp_code'] = icmp_code
        features['transport_payload_len'] = transport_payload_len
        features['time_delta'] = time_delta
        features['payload_entropy'] = pay_entropy
        features['payload_zero_ratio'] = pay_zero_ratio
        features['proto_count'] = proto_count

        return features

    except Exception as e:
        print(f"Errore nell'estrazione delle caratteristiche: {e}")
        return None


# =============================================================================
# 5) FUNZIONE PER PROCESSARE IL PCAP E SALVARE TUTTO IN CSV
# =============================================================================
def process_pcapng(file_path, class_label, output_file):
    """
    Legge un file pcapng con pyshark, estrae le 25 feature + class
    e le salva in append su un CSV (header la prima volta).
    """
    cap = pyshark.FileCapture(file_path, keep_packets=False)
    data = []
    packet_count = 0

    print(f"\tInizio elaborazione pacchetti per {file_path}")
    for packet in cap:
        features = extract_features(packet)
        if features:
            # Aggiunta colonna 'class'
            features['class'] = class_label
            data.append(features)

        packet_count += 1
        # Scrittura intermedia ogni 1000 pacchetti per limitare l'uso di RAM
        if packet_count % 1000 == 0:
            print(f"\t\t{packet_count} pacchetti elaborati...")
            pd.DataFrame(data).to_csv(
                output_file,
                mode='a',
                header=not os.path.exists(output_file),
                index=False
            )
            data.clear()

    # Scrittura finale
    if data:
        pd.DataFrame(data).to_csv(
            output_file,
            mode='a',
            header=not os.path.exists(output_file),
            index=False
        )

    cap.close()
    print(f"\tCompletato {file_path}: {packet_count} pacchetti elaborati")


# =============================================================================
# 6) MAIN
# =============================================================================
def main(input_folder, output_file):
    # Se esiste già, lo cancelliamo
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
# 7) ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Utilizzo corretto: python3 extract_packets.py <cartella_input> <file_output.csv>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_file = sys.argv[2]

    main(input_folder, output_file)
