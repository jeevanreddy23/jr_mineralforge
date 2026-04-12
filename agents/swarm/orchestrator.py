import logging
from .geometry_analyzer import GeometryAnalyzerAgent
from .searchers import ParallelSearchAgents
from .generator import FallbackGeneratorAgent
from .integrator import DataIntegratorAgent
from context_engineering.control_loop import SwarmControlLoop

logger = logging.getLogger("SwarmOrchestrator")

class SwarmOrchestrator:
    """
    Main entry point for the Team JR v2.1 Swarm.
    Implements the full recursive 'Context-Engineered' pipeline.
    """
    def __init__(self):
        self.geometry_agent = GeometryAnalyzerAgent()
        self.search_agents = ParallelSearchAgents()
        self.generator_agent = FallbackGeneratorAgent()
        self.integrator_agent = DataIntegratorAgent()
        self.loop = SwarmControlLoop(max_iterations=3, threshold=0.85)

    def run_pipeline(self, file_path: str):
        def swarm_step(context: dict):
            # 1. Geometry Analysis
            geo_res = self.geometry_agent.analyze(
                file_path, 
                iteration_buffer=context.get('bbox_buffer', 0.2) - 0.2
            )
            
            if geo_res['status'] == 'error':
                return {"status": "error", "message": geo_res['message']}
            
            bbox = geo_res['bbox']
            
            # 2. Search (Real Data)
            search_results = self.search_agents.execute({"bbox": bbox})
            
            # 3. If real data fail (or partial), trigger generator for the same bbox
            # In v2.1, we always generate synthetic as a fallback layer
            synth_cube = self.generator_agent.generate(bbox)
            
            # 4. Integrate
            final_cube = self.integrator_agent.merge(
                {"datacube_real": search_results.get('datacube'), "datacube_synth": synth_cube},
                context
            )
            
            return {
                "datacube": final_cube,
                "active_agents": ["GeometryAnalyzer", "ParallelSearchers", "FallbackGenerator"],
                "bbox": bbox
            }

        # Run the control loop
        initial_ctx = {"bbox_buffer": 0.2}
        final_result = self.loop.run(swarm_step, initial_ctx)
        
        return final_result
