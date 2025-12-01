from scapy.all import send, IP, TCP
import random
import time

# Hedef: Kendi bilgisayarın veya modemin
target_ip = "192.168.1.1" 

print(f"⚔️ {target_ip} adresine sahte trafik gönderiliyor...")

while True:
    # Rastgele portlara küçük paketler yolla (Port Scan gibi görünür)
    port = random.randint(1000, 9000)
    
    # TCP SYN paketi (Saldırıların %80'i budur)
    packet = IP(dst=target_ip)/TCP(dport=port, flags="S")
    
    send(packet, verbose=False)
    print(f"🚀 Paket yollandı -> Port {port}")
    time.sleep(0.05) # Çok hızlı yolla