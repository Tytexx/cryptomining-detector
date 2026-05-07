"""
Simple test to trigger threat detection by creating suspicious connections.
Run this in a separate terminal while monitor.py is running.
"""
import socket
import time

MINING_PORTS = [3333, 4444, 8333, 14444, 45700, 3256, 5555, 7777, 9999, 14433]

def trigger_threat():
    """Attempt connections to mining pool ports to trigger detection."""
    print("Creating suspicious connections to trigger threat detection...")
    
    for port in MINING_PORTS[:3]:  # Try first 3 ports
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            print(f"  Attempting connection to localhost:{port}...")
            sock.connect(('127.0.0.1', port))
            sock.close()
            print(f"    Connected! (or port listening)")
        except (ConnectionRefusedError, socket.timeout, OSError):
            print(f"    Connection attempt made (triggering monitor detection)")
        
        time.sleep(0.5)

if __name__ == "__main__":
    print("This test creates socket connections to mining pool ports.")
    print("Monitor.py will detect these as suspicious activity and call Claude.\n")
    
    # Keep creating connections to ensure detection
    for i in range(3):
        trigger_threat()
        time.sleep(2)
    
    print("\nTest complete. Check monitor.py output for Claude alert.")
