import http.server
import socketserver
import sys
import json
import subprocess
import socket
import urllib.request
from urllib.parse import urlparse, parse_qs

PORT = 8080

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.1)
        return s.connect_ex(('127.0.0.1', port)) == 0

def is_health_ok(port):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health", method="GET")
        with urllib.request.urlopen(req, timeout=0.2) as response:
            return response.status == 200
    except Exception:
        return False

def get_container_status(name):
    try:
        res = subprocess.run(
            ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=1.0
        )
        status = res.stdout.strip()
        return status if status else "Offline"
    except Exception:
        return "Offline"

def get_gpu_vram():

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1.0
        )
        parts = res.stdout.strip().split(",")
        if len(parts) == 2:
            used = int(parts[0].strip())
            total = int(parts[1].strip())
            return used, total
    except Exception:
        pass
    return 0, 0

def get_system_status():
    argus_status = get_container_status("argus-tinyllama-gpt")
    vanilla_status = get_container_status("vanilla-tinyllama")
    
    argus_port_bound = is_port_open(8001)
    vanilla_port_bound = is_port_open(8002)
    
    argus_ready = is_health_ok(8001)
    vanilla_ready = is_health_ok(8002)
    
    vram_used, vram_total = get_gpu_vram()
    
    return {
        "argus": {
            "container": argus_status,
            "port_bound": argus_port_bound,
            "ready": argus_ready
        },
        "vanilla": {
            "container": vanilla_status,
            "port_bound": vanilla_port_bound,
            "ready": vanilla_ready
        },
        "vram_used": vram_used,
        "vram_total": vram_total
    }


def toggle_server(target):
    try:
        if target == "vanilla":
            # 1. Kill ARGUS instantly
            subprocess.run(["docker", "kill", "argus-tinyllama-gpt"], capture_output=True)
            # 2. Start Vanilla using docker compose
            subprocess.run(["docker", "compose", "up", "-d", "vanilla-tinyllama-service"], capture_output=True)
            return {"success": True, "message": "Vanilla başlatılıyor, ARGUS durduruldu."}
        elif target == "argus":
            # 1. Kill Vanilla instantly
            subprocess.run(["docker", "kill", "vanilla-tinyllama"], capture_output=True)
            # 2. Start ARGUS using docker compose
            subprocess.run(["docker", "compose", "up", "-d", "argus-tinyllama-service"], capture_output=True)
            return {"success": True, "message": "ARGUS başlatılıyor, Vanilla durduruldu."}
        return {"success": False, "message": "Geçersiz hedef."}
    except Exception as e:
        return {"success": False, "error": str(e)}



def get_active_metrics(port):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", method="GET")
        with urllib.request.urlopen(req, timeout=0.5) as response:
            text = response.read().decode('utf-8')
            
            # Find vllm:kv_cache_usage_perc
            usage_perc = 0.0
            for line in text.splitlines():
                if line.startswith("vllm:kv_cache_usage_perc"):
                    parts = line.split()
                    if len(parts) == 2:
                        usage_perc = float(parts[1])
            
            # Find num_gpu_blocks
            num_blocks = 0
            for line in text.splitlines():
                if "num_gpu_blocks=\"" in line:
                    idx = line.find('num_gpu_blocks="')
                    if idx != -1:
                        sub = line[idx + len('num_gpu_blocks="'):]
                        end_idx = sub.find('"')
                        if end_idx != -1:
                            num_blocks = int(sub[:end_idx])
            
            return {
                "success": True,
                "kv_cache_usage_perc": usage_perc,
                "num_gpu_blocks": num_blocks
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_GET(self):
        parsed_url = urlparse(self.path)
        
        if parsed_url.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            status_data = get_system_status()
            self.wfile.write(json.dumps(status_data).encode('utf-8'))
            
        elif parsed_url.path == '/api/toggle':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            params = parse_qs(parsed_url.query)
            target = params.get('target', [None])[0]
            
            toggle_result = toggle_server(target)
            self.wfile.write(json.dumps(toggle_result).encode('utf-8'))

        elif parsed_url.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            params = parse_qs(parsed_url.query)
            port = params.get('port', ['8001'])[0]
            
            metrics_data = get_active_metrics(port)
            self.wfile.write(json.dumps(metrics_data).encode('utf-8'))
        else:
            super().do_GET()

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print("=" * 75)
        print(" ⚡ ARGUS-vLLM İnteraktif Canlı Arayüzü Başlatıldı! ⚡")
        print("=" * 75)
        print(f" 👉 Tarayıcınızda şu adresi açın: http://localhost:{PORT}/benchmarks/ui.html")
        print("=" * 75)
        print("Sunucuyu kapatmak için CTRL+C tuşlarına basın.")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nSunucu kapatılıyor...")
            sys.exit(0)

if __name__ == "__main__":
    run_server()
