import psutil
import joblib
import pandas as pd
import time
import os
import sys
import traceback

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if os.getenv('GEMINI_API_KEY'):
    # Avoid the Google client library preferring GOOGLE_API_KEY when both are present.
    os.environ.pop('GOOGLE_API_KEY', None)

try:
    from google import genai
except ImportError:
    genai = None

rf = joblib.load('cryptomining_detector.pkl')
print("✓ Model loaded!")
if genai:
    if os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'):
        print("✓ Gemini API configured")
    else:
        print("⚠ Gemini API client installed, but no GEMINI_API_KEY or GOOGLE_API_KEY is set")
else:
    print("⚠ Gemini API not available (install: pip install google-genai)")
print("Starting monitor... Press Ctrl+C to stop")
print("="*50)

MINING_PORTS = [3333, 4444, 8333, 14444, 45700,
                3256, 5555, 7777, 9999, 14433]

def collect_metrics():
    cpu = psutil.cpu_times_percent(interval=1)
    cpu_total = psutil.cpu_percent()

    load = psutil.getloadavg()

    procs = list(psutil.process_iter(['status', 'num_threads']))
    statuses = [p.info['status'] for p in procs]
    threads = sum(p.info['num_threads'] for p in procs if p.info['num_threads'])

    disk1 = psutil.disk_io_counters()
    time.sleep(1)
    disk2 = psutil.disk_io_counters()
    write_rate = disk2.write_bytes - disk1.write_bytes

    net = psutil.net_io_counters(pernic=True)
    lo = net.get('lo', None)

    metrics = {
        'cpu_iowait':             getattr(cpu, 'iowait', 0),
        'cpu_nice':               getattr(cpu, 'nice', 0),
        'cpu_system':             cpu.system,
        'cpu_total':              cpu_total,
        'cpu_user':               cpu.user,
        'diskio_sda_write_bytes': write_rate,
        'load_cpucore':           psutil.cpu_count(),
        'load_min1':              load[0],
        'load_min15':             load[2],
        'load_min5':              load[1],
        'network_lo_rx':          lo.bytes_recv if lo else 0,
        'processcount_running':   statuses.count('running'),
        'processcount_sleeping':  statuses.count('sleeping'),
        'processcount_thread':    threads,
        'processcount_total':     len(procs),
    }

    return metrics

def predict(metrics):
    df = pd.DataFrame([metrics])
    probability = rf.predict_proba(df)[0]
    return probability

def check_mining_connections():
    suspicious = []
    try:
        connections = psutil.net_connections()
        for conn in connections:
            if conn.raddr and conn.raddr.port in MINING_PORTS:
                suspicious.append({
                    'ip': conn.raddr.ip,
                    'port': conn.raddr.port,
                    'status': conn.status,
                    'pid': conn.pid
                })
    except:
        pass
    return suspicious

def get_process_name(pid):
    try:
        process = psutil.Process(pid)
        return process.name()
    except:
        return "Unknown"

def get_combined_status(attack_prob, suspicious_connections):
    network_threat = len(suspicious_connections) > 0

    if network_threat:
        return "Mining Pool Connection Detected!", "NETWORK"

    # ML based detection
    elif attack_prob >= 80:
        return "High Probability of Cryptomining Attack!", "ML"
    elif attack_prob >= 60:
        return "Probablity of Cryptomining Attack", "ML"
    else:
        return "No Threat Detected", "NORMAL"


def generate_gemini_alert(metrics, attack_prob, suspicious_connections, detection_type):
    """Generate a Gemini-based alert summary for suspicious activity."""
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key or genai is None:
        return None

    client = genai.Client(api_key=api_key)
    model_candidates = [
        os.getenv('GEMINI_MODEL'),
        'gemini-2.5-flash',
        'gemini-2.0-flash',
        'gemini-1.5-flash-002',
        'gemini-1.5-flash',
    ]
    model_candidates = [name for name in model_candidates if name]

    # Build evidence for Gemini
    evidence = []
    if detection_type == "NETWORK" and suspicious_connections:
        evidence.append(f"Found {len(suspicious_connections)} connection(s) to known mining pools")
        for conn in suspicious_connections:
            evidence.append(f"  - {conn['ip']}:{conn['port']} (status: {conn['status']})")
    
    if attack_prob >= 60:
        evidence.append(f"ML model confidence: {attack_prob:.1f}%")
        evidence.append(f"Anomalous metrics:")
        evidence.append(f"  - CPU total: {metrics['cpu_total']:.1f}%")
        evidence.append(f"  - Load 1min: {metrics['load_min1']:.2f}")
        evidence.append(f"  - Processes: {metrics['processcount_total']}")
        evidence.append(f"  - Threads: {metrics['processcount_thread']}")

    evidence_text = '\n'.join(evidence) if evidence else "Generic anomaly detected"

    prompt = f"""
You are a security analyst writing a brief alert for a system administrator.

Based on this live detection, provide a 2-3 sentence alert that:
1. Clearly states the threat level and what was detected
2. Explains why this activity is suspicious
3. Suggests one immediate action to take

Keep it actionable and brief. Do not speculate.

Detection Evidence:
{evidence_text}

Write the alert now:
""".strip()

    last_error = None
    for model_name in model_candidates:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            try:
                text = response.text.strip()
                if text:
                    return text
            except Exception:
                print("[Gemini] unexpected response format", file=sys.stderr)
                print(response, file=sys.stderr)
                return None
        except Exception as e:
            last_error = e
            print(f"[Gemini] model '{model_name}' failed: {e}", file=sys.stderr)
            continue

    # Surface detailed error information for debugging
    if last_error is not None:
        print(f"[Gemini] API call failed: {last_error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Also print the prompt (without API key) to help debugging
        try:
            print("[Gemini] Prompt:\n" + prompt, file=sys.stderr)
        except Exception:
            pass
    return None

# Main loop
try:
    while True:
        metrics = collect_metrics()
        probability = predict(metrics)
        attack_prob = probability[1] * 100
        suspicious_connections = check_mining_connections()
        status, detection_type = get_combined_status(
            attack_prob, suspicious_connections)

        os.system('clear')
        print("="*55)
        print("     CRYPTOMINING DETECTOR - LIVE MONITOR")
        print("="*55)

        print(f"\n  {status}")
        print(f"\n  ML Confidence:     {attack_prob:.1f}%")
        print(f"  Detection Method:  {detection_type}")

        # Generate Gemini alert if threat detected
        if detection_type in ["NETWORK", "ML"] and (len(suspicious_connections) > 0 or attack_prob >= 60):
            gemini_alert = generate_gemini_alert(metrics, attack_prob, suspicious_connections, detection_type)
            if gemini_alert:
                print(f"\n--- AI Security Analysis ---")
                print(f"  {gemini_alert}")
            else:
                print(f"\n--- Security Analysis ---")
                if detection_type == "NETWORK":
                    print(f"  ⚠ Suspicious mining pool connections detected.")
                else:
                    print(f"  ⚠ System metrics indicate potential cryptomining activity.")

        print(f"\n--- System Metrics ---")
        print(f"  CPU Total:     {metrics['cpu_total']:.1f}%")
        print(f"  CPU User:      {metrics['cpu_user']:.1f}%")
        print(f"  CPU System:    {metrics['cpu_system']:.1f}%")
        print(f"  Load 1min:     {metrics['load_min1']:.2f}")
        print(f"  Load 5min:     {metrics['load_min5']:.2f}")
        print(f"  Load 15min:    {metrics['load_min15']:.2f}")
        print(f"  Processes:     {metrics['processcount_total']}")
        print(f"  Threads:       {metrics['processcount_thread']}")
        print(f"  Disk Writes:   {metrics['diskio_sda_write_bytes']}")

        print(f"\n--- Network Analysis ---")
        if suspicious_connections:
            print(f"{len(suspicious_connections)} suspicious connection(s) found!")
            for conn in suspicious_connections:
                name = get_process_name(conn['pid'])
                print(f"  Process: {name}")
                print(f"  Remote:  {conn['ip']}:{conn['port']}")
                print(f"  Status:  {conn['status']}")
        else:
            print("No suspicious connections found")

        print(f"\n  Updating every 5 seconds... Ctrl+C to stop")
        print("="*55)

        time.sleep(5)

except KeyboardInterrupt:
    print("\n✓ Monitor stopped.")