import psutil
import joblib
import pandas as pd
import time
import os

rf = joblib.load('cryptomining_detector.pkl')
print("✓ Model loaded!")
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