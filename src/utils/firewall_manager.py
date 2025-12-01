import os
import platform
import subprocess

# Güvenli Liste: Kendimizi veya modemi yanlışlıkla engellemeyelim
WHITELIST = ["127.0.0.1", "localhost", "192.168.1.1", "0.0.0.0"]

def get_os():
    return platform.system()

def block_ip(ip_address):
    """
    Verilen IP adresini işletim sistemi seviyesinde engeller.
    """
    if ip_address in WHITELIST:
        print(f"⚠️  UYARI: {ip_address} güvenli listede, engellenemez!")
        return False

    os_name = get_os()
    
    try:
        if os_name == "Windows":
            # Windows Firewall (Netsh)
            rule_name = f"Block_AI_{ip_address}"
            
            # Zaten var mı kontrol et
            check_cmd = f"netsh advfirewall firewall show rule name=\"{rule_name}\""
            if subprocess.call(check_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                return True # Zaten engelli

            # Ekle
            command = f"netsh advfirewall firewall add rule name=\"{rule_name}\" dir=in action=block remoteip={ip_address}"
            os.system(command)
            print(f"🚫 [WINDOWS] {ip_address} güvenlik duvarı tarafından engellendi!")
            
        elif os_name == "Linux":
            # Linux IPTables
            check_cmd = f"iptables -C INPUT -s {ip_address} -j DROP"
            if subprocess.call(check_cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                return True
                
            command = f"iptables -A INPUT -s {ip_address} -j DROP"
            os.system(command)
            print(f"🚫 [LINUX] {ip_address} güvenlik duvarı tarafından engellendi!")
            
        return True

    except Exception as e:
        print(f"❌ Engelleme Hatası: {e}")
        return False