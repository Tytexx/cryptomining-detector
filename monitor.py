import psutil
import joblib
import pandas as pd
import time
import os

# Load model
rf = joblib.load('cryptomining_detector.pkl')
print("✓ Model loaded!")
print("Starting monitor... Press Ctrl+C to stop")
print("="*50)

def get_status(probability):
    attack_prob = probability[1] * 100
    
    if attack_prob >= 80:
        return "🚨 CRITICAL - High Probability Cryptomining Attack!", attack_prob
    elif attack_prob >= 60:
        return "⚠️  WARNING - Possible Cryptomining Attack", attack_prob
    else:
        return "✅ NORMAL - No Threat Detected", attack_prob

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
    prediction = rf.predict(df)[0]
    probability = rf.predict_proba(df)[0]
    return prediction, probability

# Main loop
try:
   while True:
    metrics = collect_metrics()
    prediction, probability = predict(metrics)
    status, confidence = get_status(probability)
    
    os.system('clear')
    print("="*50)
    print("  CRYPTOMINING DETECTOR - LIVE MONITOR")
    print("="*50)
    print(f"\n  {status}")
    print(f"  Attack Probability: {confidence:.1f}%")
    print(f"\n--- Current Metrics ---")
    print(f"CPU Total:     {metrics['cpu_total']:.1f}%")
    print(f"CPU User:      {metrics['cpu_user']:.1f}%")
    print(f"CPU System:    {metrics['cpu_system']:.1f}%")
    print(f"Load 1min:     {metrics['load_min1']:.2f}")
    print(f"Load 5min:     {metrics['load_min5']:.2f}")
    print(f"Load 15min:    {metrics['load_min15']:.2f}")
    print(f"Processes:     {metrics['processcount_total']}")
    print(f"Threads:       {metrics['processcount_thread']}")
    print(f"Disk Writes:   {metrics['diskio_sda_write_bytes']}")
    
    time.sleep(5)

except KeyboardInterrupt:
    print("\n Monitor stopped.")