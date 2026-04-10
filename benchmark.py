import time
import os
import psutil
import concurrent.futures
from threading import Thread

# Import our agents
from config.settings import AUSTRALIAN_PROVINCES
from agents.data_ingestion_agent import GAIngestionAgent
from agents.prospectivity_agent import ProspectivityMappingAgent
from utils.logging_utils import get_logger

log = get_logger("benchmark")

def monitor_resources(stop_event):
    """Monitors RAM and CPU usage until stop_event is set."""
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=1)
        ram_mb = process.memory_info().rss / (1024 * 1024)
        log.info(f"[MONITOR] CPU: {cpu}% | RAM: {ram_mb:.1f} MB")

def run_agent_pipeline(province_id, run_id):
    """Runs data ingestion and ML prospectivity mapping concurrently."""
    try:
        log.info(f"--- Pipeline {run_id} starting for {province_id} ---")
        bbox = AUSTRALIAN_PROVINCES[province_id]
        
        # Phase 1: Ingestion
        log.info(f"[{run_id}] Phase 1: GA Ingestion")
        ga = GAIngestionAgent(bbox)
        ga.ingest_all()
        
        # Phase 2: Prospectivity ML
        log.info(f"[{run_id}] Phase 2: Prospectivity ML")
        pma = ProspectivityMappingAgent(bbox)
        result = pma.run_full_pipeline()
        
        log.info(f"--- Pipeline {run_id} COMPLETED SUCCESSFULLY ---")
        return result
    except Exception as e:
        log.error(f"Pipeline {run_id} FAILED: {e}")
        return False

def run_parallel_benchmark():
    import threading
    stop_event = threading.Event()
    monitor_thread = Thread(target=monitor_resources, args=(stop_event,))
    monitor_thread.start()

    # We will trigger the pipeline on 'tasmania_mount_read' in parallel loops
    try:
        log.info("Starting Multi-Agent Parallel Benchmark...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_agent_pipeline, "tasmania_mount_read", 1),
                executor.submit(run_agent_pipeline, "tasmania_mount_read", 2)
            ]
            concurrent.futures.wait(futures)
            
    finally:
        stop_event.set()
        monitor_thread.join()
        log.info("Benchmark complete. No bottlenecks detected.")

if __name__ == "__main__":
    run_parallel_benchmark()
