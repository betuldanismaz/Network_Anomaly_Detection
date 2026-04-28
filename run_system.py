#!/usr/bin/env python3
"""
Network Anomaly Detection System - Automated Launcher
Orchestrates Docker, Kafka Consumer, Producer, and Dashboard in separate terminals.
"""

import os
import sys
import time
import platform
import subprocess

# ---------------------------------------------------------------------------
# ANSI COLORS
# ---------------------------------------------------------------------------
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BOLD = '\033[1m'
RESET = '\033[0m'

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable


def print_banner():
    """Print startup banner."""
    print(f"\n{BOLD}{CYAN}╔{'═'*68}╗{RESET}")
    print(f"{BOLD}{CYAN}║{' '*10}🛡️  NETWORK ANOMALY DETECTION SYSTEM LAUNCHER{' '*10}║{RESET}")
    print(f"{BOLD}{CYAN}║{' '*20}Automated Multi-Service Startup{' '*20}║{RESET}")
    print(f"{BOLD}{CYAN}╚{'═'*68}╝{RESET}\n")


def check_os():
    """Verify running on Windows."""
    current_os = platform.system()
    print(f"{CYAN}🖥️  Operating System: {current_os}{RESET}")
    
    if current_os != "Windows":
        print(f"{RED}❌ ERROR: This script is designed for Windows only.{RESET}")
        print(f"{YELLOW}   Current OS: {current_os}{RESET}")
        print(f"{YELLOW}   Please use manual startup or adapt the script.{RESET}")
        sys.exit(1)
    
    print(f"{GREEN}✅ Windows detected - proceeding with automation{RESET}\n")


def check_docker_available():
    """Check if Docker is installed and available."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def start_docker_infrastructure():
    """Start Docker Compose services (Zookeeper & Kafka)."""
    print(f"{BOLD}{MAGENTA}{'─'*70}{RESET}")
    print(f"{BOLD}{MAGENTA}STEP 1: Starting Docker Infrastructure{RESET}")
    print(f"{BOLD}{MAGENTA}{'─'*70}{RESET}\n")
    
    # Check if Docker is available
    if not check_docker_available():
        print(f"{YELLOW}⚠️  Docker is not installed or not in PATH{RESET}")
        print(f"{CYAN}╭{'─'*68}╮{RESET}")
        print(f"{CYAN}│{BOLD} DOCKER KURULUM TALİMATLARI:{RESET}{' '*39}{CYAN}│{RESET}")
        print(f"{CYAN}│{' '*68}│{RESET}")
        print(f"{CYAN}│  1. Docker Desktop for Windows'u indirin:{' '*26}│{RESET}")
        print(f"{CYAN}│     https://www.docker.com/products/docker-desktop{' '*17}│{RESET}")
        print(f"{CYAN}│{' '*68}│{RESET}")
        print(f"{CYAN}│  2. Kurun ve bilgisayarı yeniden başlatın{' '*26}│{RESET}")
        print(f"{CYAN}│{' '*68}│{RESET}")
        print(f"{CYAN}│  3. Docker Desktop'ı çalıştırın{' '*36}│{RESET}")
        print(f"{CYAN}│{' '*68}│{RESET}")
        print(f"{CYAN}│  4. Bu scripti tekrar çalıştırın: python run_system.py{' '*13}│{RESET}")
        print(f"{CYAN}╰{'─'*68}╯{RESET}\n")
        
        print(f"{RED}❌ Docker olmadan sistem çalışmaz. Lütfen Docker'ı kurun.{RESET}")
        print(f"{YELLOW}Program sonlandırılıyor...{RESET}\n")
        sys.exit(1)
    
    # Check if docker-compose.yml exists
    docker_compose_path = os.path.join(PROJECT_ROOT, "docker-compose.yml")
    if not os.path.exists(docker_compose_path):
        print(f"{RED}❌ ERROR: docker-compose.yml not found!{RESET}")
        print(f"   Expected: {docker_compose_path}")
        sys.exit(1)
    
    print(f"{CYAN}📦 Launching Docker Compose...{RESET}")
    
    try:
        # Run docker-compose up -d
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"{GREEN}✅ Docker containers started successfully{RESET}")
            if result.stdout:
                # Print first line of output
                output_line = result.stdout.strip().split('\n')[0]
                print(f"{CYAN}   {output_line}{RESET}")
        else:
            print(f"{YELLOW}⚠️  Docker command completed with warnings{RESET}")
            if result.stderr:
                print(f"{YELLOW}   Error: {result.stderr.strip()[:200]}{RESET}")
        
        # Wait for Kafka to warm up
        print(f"\n{YELLOW}⏳ Waiting 10 seconds for Kafka to warm up...{RESET}")
        for i in range(10, 0, -1):
            print(f"   {i} seconds remaining...", end="\r")
            time.sleep(1)
        print(f"{GREEN}✅ Kafka warmup complete{RESET}" + " " * 30 + "\n")
        
    except subprocess.TimeoutExpired:
        print(f"{RED}❌ ERROR: Docker command timed out{RESET}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"{RED}❌ ERROR: docker-compose command not found{RESET}")
        print(f"{YELLOW}   Try: docker compose up -d (newer Docker versions){RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}❌ ERROR: Failed to start Docker{RESET}")
        print(f"   {e}")
        sys.exit(1)


def launch_in_new_terminal(command, title, working_dir=None):
    """
    Launch a command in a new Windows terminal window.
    
    Args:
        command: Command to execute (string)
        title: Window title
        working_dir: Working directory (defaults to PROJECT_ROOT)
    
    Returns:
        subprocess.Popen object
    """
    if working_dir is None:
        working_dir = PROJECT_ROOT
    
    # Windows command pattern: start cmd /k "command"
    # /k keeps the window open after command execution
    full_command = f'start "{title}" cmd /k "cd /d {working_dir} && {command}"'
    
    try:
        process = subprocess.Popen(
            full_command,
            shell=True,
            cwd=working_dir
        )
        return process
    except Exception as e:
        print(f"{RED}⚠️  Failed to launch {title}: {e}{RESET}")
        return None


def launch_services():
    """Launch Consumer, Dashboard, and Producer in separate terminals."""
    print(f"{BOLD}{MAGENTA}{'─'*70}{RESET}")
    print(f"{BOLD}{MAGENTA}STEP 2: Launching Background Services{RESET}")
    print(f"{BOLD}{MAGENTA}{'─'*70}{RESET}\n")
    
    services = []
    
    # 1. KAFKA CONSUMER
    print(f"{CYAN}🔄 Launching Kafka Consumer...{RESET}")
    consumer_process = launch_in_new_terminal(
        f'"{PYTHON_EXE}" src\\kafka_consumer.py',
        "NIDS - Kafka Consumer",
        PROJECT_ROOT
    )
    if consumer_process:
        services.append(("Consumer", consumer_process))
        print(f"{GREEN}✅ Consumer launched in new terminal{RESET}")
        time.sleep(2)  # Brief pause to avoid terminal spam
    
    # 2. STREAMLIT DASHBOARD
    print(f"{CYAN}📊 Launching Streamlit Dashboard...{RESET}")
    dashboard_process = launch_in_new_terminal(
        f'"{PYTHON_EXE}" -m streamlit run src\\dashboard\\app.py',
        "NIDS - Dashboard",
        PROJECT_ROOT
    )
    if dashboard_process:
        services.append(("Dashboard", dashboard_process))
        print(f"{GREEN}✅ Dashboard launched in new terminal{RESET}")
        print(f"{CYAN}   URL: http://localhost:8501{RESET}")
        time.sleep(2)
    
    print(f"\n{BOLD}{MAGENTA}{'─'*70}{RESET}")
    print(f"{BOLD}{MAGENTA}STEP 3: Launching Traffic Producer{RESET}")
    print(f"{BOLD}{MAGENTA}{'─'*70}{RESET}\n")
    
    # 3. LIVE BRIDGE (PRODUCER)
    print(f"{CYAN}📡 Launching Live Bridge Producer...{RESET}")
    producer_process = launch_in_new_terminal(
        f'"{PYTHON_EXE}" src\\live_bridge.py',
        "NIDS - Producer (Live Bridge)",
        PROJECT_ROOT
    )
    if producer_process:
        services.append(("Producer", producer_process))
        print(f"{GREEN}✅ Producer launched in new terminal{RESET}")
    
    return services


def print_system_status(services):
    """Print final system status and instructions."""
    print(f"\n{BOLD}{GREEN}{'═'*70}{RESET}")
    print(f"{BOLD}{GREEN}🚀 SYSTEM STARTUP COMPLETE{RESET}")
    print(f"{BOLD}{GREEN}{'═'*70}{RESET}\n")
    
    print(f"{CYAN}Active Services:{RESET}")
    print(f"   {GREEN}✓{RESET} Docker (Zookeeper + Kafka)")
    for service_name, _ in services:
        print(f"   {GREEN}✓{RESET} {service_name}")
    
    print(f"\n{CYAN}Service Windows:{RESET}")
    print(f"   📊 Dashboard:  http://localhost:8501")
    print(f"   🔄 Consumer:   Check 'NIDS - Kafka Consumer' terminal")
    print(f"   📡 Producer:   Check 'NIDS - Producer' terminal")
    print(f"   🐳 Docker:     docker ps")
    
    print(f"\n{YELLOW}{'─'*70}{RESET}")
    print(f"{YELLOW}⚠️  ÖNEMLİ NOTLAR:{RESET}")
    print(f"   • Her servis kendi terminal penceresinde çalışıyor")
    print(f"   • Durdurmak için: Her terminalde CTRL+C yapın")
    print(f"   • Docker'ı durdurmak için: docker-compose down")
    print(f"   • Logları ilgili terminal pencerelerinden izleyin")
    print(f"{YELLOW}{'─'*70}{RESET}\n")
    
    print(f"{BOLD}{CYAN}Sistem şimdi gerçek zamanlı ağ trafiğini işliyor!{RESET}")
    print(f"{CYAN}Bu pencereyi kapatabilirsiniz (servisler çalışmaya devam eder){RESET}\n")


def main():
    """Main orchestration function."""
    try:
        print_banner()
        check_os()
        start_docker_infrastructure()
        services = launch_services()
        print_system_status(services)
        
        # Keep launcher alive to show status
        print(f"{CYAN}Launcher aktif... (servisler bağımsız çalışıyor){RESET}")
        print(f"{YELLOW}Bu pencereyi kapatabilirsiniz - servisler çalışmaya devam eder.{RESET}\n")
        
        # Optional: Keep script running
        try:
            print(f"{CYAN}Çıkmak için CTRL+C yapın...{RESET}")
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print(f"\n{GREEN}✅ Launcher kapatılıyor (servisler hala çalışıyor){RESET}")
            print(f"{CYAN}Servisler kendi terminal pencerelerinde aktif{RESET}\n")
    
    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠️  Launcher interrupted{RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{RED}❌ FATAL ERROR: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
