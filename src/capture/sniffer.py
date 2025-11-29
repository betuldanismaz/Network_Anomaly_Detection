# listens to the network, captures 10 IP packets, and prints their source, destination, and size.


from scapy.all import sniff, IP

def packet_callback(packet):
    if packet.haslayer(IP):
        src = packet[IP].src
        dst = packet[IP].dst
        print(f"Paket Yakalandı: {src} --> {dst} | Uzunluk: {len(packet)} byte")

if __name__ == "__main__":
    print("Ağ dinleniyor... (Durdurmak için Ctrl+C)")
    # 10 paket yakalayıp duracak şekilde ayarlı
    sniff(count=10, prn=packet_callback)