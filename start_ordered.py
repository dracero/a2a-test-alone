# start_ordered.py
# Native Python script to start BeeAI Ecosystem agents and frontend in order.
# Works on Windows, macOS, and Linux without bash/WSL or nc.

import sys
import os
import time
import socket
import subprocess
import signal

# Reconfigure encoding for Windows console to prevent Unicode/emoji encoding errors
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

processes = []

def cleanup(sig=None, frame=None):
    print("\n🛑 Stopping all services...")
    for p in processes:
        if p.poll() is None:
            try:
                if sys.platform == 'win32':
                    # Kill the entire process tree on Windows to ensure child node/python processes die
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    p.kill()
            except Exception:
                pass
    sys.exit(0)

# Handle Ctrl+C and exit signals
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.connect(('127.0.0.1', port))
            return True
        except Exception:
            return False

def wait_for_port(port, name):
    print("--------------------------------------------------")
    print(f"⏳ Waiting for {name} on port {port}...")
    while not is_port_open(port):
        time.sleep(2)
    print(f"✅ {name} is ready!")

def run_npm_cmd(cmd, env_vars=None):
    # Use shell=True to support npm commands resolving in Windows/Linux PATH
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    p = subprocess.Popen(cmd, shell=True, env=env)
    processes.append(p)
    return p

def main():
    os.environ["NODE_OPTIONS"] = "--no-deprecation"
    print("🚀 Starting BeeAI Ecosystem in order...")
    
    # 1. Start Priority Agent (Multimodal)
    print("Step 1: Starting Priority Agent (10003)...")
    run_npm_cmd("npm run dev:agent:multimodal", {"LANGCHAIN_PROJECT": "a2a-multimodal-tutor"})
    wait_for_port(10003, "Multimodal Agent")
    
    # 2. Start Remaining Agents (Images & Medical)
    print("\nStep 2: Starting Remaining Agents (10001 & 10002)...")
    run_npm_cmd("npm run dev:agent:images", {"LANGCHAIN_PROJECT": "a2a-image-generator"})
    run_npm_cmd("npm run dev:agent:medical", {"LANGCHAIN_PROJECT": "a2a-medical-assistant"})
    wait_for_port(10001, "Images Agent")
    wait_for_port(10002, "Medical Agent")
    
    # 3. Start Orchestrator
    print("\nStep 3: Starting Orchestrator (Backend)...")
    run_npm_cmd("npm run dev:backend", {"LANGCHAIN_PROJECT": "a2a-orchestrator"})
    wait_for_port(12000, "Orchestrator")
    
    # 4. Start Frontend
    print("\nStep 4: Starting Frontend...")
    print("--------------------------------------------------")
    print("Frontend will be available at http://localhost:3000")
    print("Press Ctrl+C to stop everything.")
    print("--------------------------------------------------")
    
    # Start frontend in foreground and wait for it
    p_frontend = run_npm_cmd("npm run dev:frontend")
    try:
        p_frontend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == '__main__':
    main()
