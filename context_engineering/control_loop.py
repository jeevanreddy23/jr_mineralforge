from typing import Callable, Dict, Any, List
import logging
from .field_resonance_measure import FieldResonanceMeasure

logger = logging.getLogger("SwarmControlLoop")

class SwarmControlLoop:
    """
    Team JR v2.1 Execution Controller.
    Implements a recursive feedback loop that expands context (BBOX)
    if the data resonance score is below threshold.
    """
    
    def __init__(self, max_iterations: int = 3, threshold: float = 0.85):
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.measure = FieldResonanceMeasure()

    def run(self, swarm_execute_fn: Callable, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the provided swarm function recursively.
        """
        current_context = initial_context.copy()
        current_context.setdefault('iteration', 0)
        current_context.setdefault('bbox_buffer', 0.2) # Initial 20%
        current_context.setdefault('history', [])
        
        last_result = None
        
        while current_context['iteration'] < self.max_iterations:
            idx = current_context['iteration']
            logger.info(f"[Team JR Swarm] Cycle {idx+1} starting...")
            
            # 1. Run the Swarm Logic
            result = swarm_execute_fn(current_context)
            
            # 2. Extract Data and Measure Resonance
            datacube = result.get('datacube')
            resonance = self.measure.calculate_resonance(datacube, current_context)
            
            result['resonance_score'] = resonance
            result['resonance_status'] = self.measure.get_status_label(resonance)
            
            # Update history for dashboard
            current_context['history'].append({
                "iteration": idx + 1,
                "resonance": resonance,
                "status": result['resonance_status'],
                "agents": result.get('active_agents', [])
            })
            
            last_result = result
            
            # Check exit
            if resonance >= self.threshold:
                logger.info(f"[Team JR Swarm] Convergence at {resonance:.2f}")
                break
                
            # 3. Recursive Context Expansion
            current_context['iteration'] += 1
            current_context['bbox_buffer'] += 0.15 # Expand by 15% each try
            logger.warning(f"[Team JR Swarm] Low Resonance. Re-configuring context with buffer={current_context['bbox_buffer']:.2f}")

        return last_result
