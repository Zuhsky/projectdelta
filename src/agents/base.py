import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from ..utils.ollama_client import RunPodOllamaClient, OllamaResponse
from ..utils.logger import get_logger, log_agent_start, log_agent_complete, log_gpu_status
from ..core.config import config

console = Console()

class GPUOptimizedAgent(ABC):
    """GPU-optimized base class for all AI agents."""
    
    def __init__(self, ollama_client: RunPodOllamaClient, model: str, agent_name: str):
        self.ollama_client = ollama_client
        self.model = model
        self.agent_name = agent_name
        self.console = Console()
        self.logger = get_logger(self.__class__.__name__)
        self.execution_metrics = []
        self.gpu_info = None
    
    @abstractmethod
    async def run_async(self, *args, **kwargs) -> Any:
        """Async execution method - must be implemented by subclasses."""
        pass
    
    def run(self, *args, **kwargs) -> Any:
        """Synchronous wrapper for async execution."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.run_async(*args, **kwargs))
        finally:
            loop.close()
    
    async def execute_with_monitoring(self, func, *args, **kwargs) -> Any:
        """Execute function with comprehensive monitoring and error handling."""
        start_time = time.time()
        start_resources = self.ollama_client.monitor_resources()
        self.gpu_info = start_resources.get("gpu_info", {})
        
        # Log agent start
        log_agent_start(self.logger, self.agent_name, self.model)
        log_gpu_status(self.logger, self.gpu_info)
        
        try:
            # Execute the agent's main function
            result = await func(*args, **kwargs)
            
            # Record successful execution metrics
            end_time = time.time()
            end_resources = self.ollama_client.monitor_resources()
            execution_time = end_time - start_time
            
            self.execution_metrics.append({
                "agent": self.agent_name,
                "model": self.model,
                "execution_time": execution_time,
                "success": True,
                "start_resources": start_resources,
                "end_resources": end_resources,
                "gpu_info": self.gpu_info,
                "timestamp": time.time()
            })
            
            # Log completion
            log_agent_complete(self.logger, self.agent_name, execution_time, True)
            
            return result
            
        except Exception as e:
            # Record failed execution metrics
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.execution_metrics.append({
                "agent": self.agent_name,
                "model": self.model,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "gpu_info": self.gpu_info,
                "timestamp": time.time()
            })
            
            # Log error
            self.logger.error(f"{self.agent_name} failed after {execution_time:.2f}s: {e}")
            log_agent_complete(self.logger, self.agent_name, execution_time, False)
            
            raise
    
    def load_prompt(self, prompt_file: str) -> str:
        """Load prompt from file with comprehensive error handling."""
        try:
            prompt_path = f"prompts/{prompt_file}"
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_content = f.read()
            
            self.logger.debug(f"Loaded prompt from {prompt_path} ({len(prompt_content)} characters)")
            return prompt_content
            
        except FileNotFoundError:
            self.logger.error(f"Prompt file not found: {prompt_file}")
            console.print(f"[red]❌ Prompt file not found: {prompt_file}[/red]")
            return ""
        except Exception as e:
            self.logger.error(f"Error loading prompt {prompt_file}: {e}")
            console.print(f"[red]❌ Error loading prompt: {e}[/red]")
            return ""
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data with detailed error reporting."""
        if not isinstance(data, dict):
            self.logger.error("Input data must be a dictionary")
            return False
        
        # Add specific validation logic here based on agent type
        return True
    
    def handle_error(self, error: Exception, context: str = "") -> None:
        """Centralized error handling with GPU context."""
        error_context = f"{self.agent_name}: {context}" if context else self.agent_name
        self.logger.error(f"{error_context} - Error: {error}")
        
        # Log GPU state during error
        if self.gpu_info:
            self.logger.error(f"GPU state during error: {self.gpu_info}")
        
        console.print(f"[red]❌ {error_context} error: {error}[/red]")
    
    async def generate_with_retry(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        max_retries: int = None,
        **kwargs
    ) -> Optional[OllamaResponse]:
        """Generate response with automatic retry and GPU optimization."""
        if max_retries is None:
            max_retries = config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.ollama_client.generate_async(
                    self.model, 
                    prompt, 
                    system=system,
                    **kwargs
                )
                
                if response:
                    self.logger.info(f"Generated response in {response.response_time:.2f}s")
                    return response
                else:
                    raise Exception("Empty response from model")
                    
            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All {max_retries + 1} attempts failed: {e}")
                    raise
        
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report for this agent."""
        if not self.execution_metrics:
            return {
                "agent": self.agent_name,
                "model": self.model,
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "total_execution_time": 0.0,
                "fastest_execution": 0.0,
                "slowest_execution": 0.0,
                "gpu_utilization": {}
            }
        
        successful_runs = [m for m in self.execution_metrics if m.get("success")]
        failed_runs = [m for m in self.execution_metrics if not m.get("success")]
        
        # Calculate GPU utilization
        gpu_utilization = {}
        if self.gpu_info:
            gpu_utilization = {
                "gpu_name": self.gpu_info.get("name", "Unknown"),
                "memory_usage": self.gpu_info.get("memory_used", 0) / self.gpu_info.get("memory_total", 1) * 100,
                "temperature": self.gpu_info.get("temperature", 0),
                "load": self.gpu_info.get("load", 0) * 100
            }
        
        return {
            "agent": self.agent_name,
            "model": self.model,
            "total_executions": len(self.execution_metrics),
            "successful_executions": len(successful_runs),
            "failed_executions": len(failed_runs),
            "success_rate": len(successful_runs) / len(self.execution_metrics) if self.execution_metrics else 0,
            "average_execution_time": sum(m["execution_time"] for m in successful_runs) / len(successful_runs) if successful_runs else 0,
            "total_execution_time": sum(m["execution_time"] for m in self.execution_metrics),
            "fastest_execution": min(m["execution_time"] for m in self.execution_metrics) if self.execution_metrics else 0,
            "slowest_execution": max(m["execution_time"] for m in self.execution_metrics) if self.execution_metrics else 0,
            "gpu_utilization": gpu_utilization,
            "recent_errors": [m.get("error") for m in failed_runs[-5:]]  # Last 5 errors
        }
    
    def display_performance_summary(self):
        """Display performance summary in a rich format."""
        report = self.get_performance_report()
        
        if report["total_executions"] == 0:
            console.print(f"[yellow]No execution data available for {self.agent_name}[/yellow]")
            return
        
        # Create performance panel
        performance_text = f"""
[bold]Agent:[/bold] {report['agent']}
[bold]Model:[/bold] {report['model']}
[bold]Total Executions:[/bold] {report['total_executions']}
[bold]Success Rate:[/bold] {report['success_rate']:.1%}
[bold]Average Time:[/bold] {report['average_execution_time']:.2f}s
[bold]Total Time:[/bold] {report['total_execution_time']:.2f}s
[bold]Fastest:[/bold] {report['fastest_execution']:.2f}s
[bold]Slowest:[/bold] {report['slowest_execution']:.2f}s
        """
        
        if report["gpu_utilization"]:
            gpu_text = f"""
[bold]GPU:[/bold] {report['gpu_utilization']['gpu_name']}
[bold]Memory Usage:[/bold] {report['gpu_utilization']['memory_usage']:.1f}%
[bold]Temperature:[/bold] {report['gpu_utilization']['temperature']:.1f}°C
[bold]Load:[/bold] {report['gpu_utilization']['load']:.1f}%
            """
            performance_text += gpu_text
        
        if report["recent_errors"]:
            error_text = "\n[bold]Recent Errors:[/bold]\n"
            for error in report["recent_errors"]:
                error_text += f"• {error}\n"
            performance_text += error_text
        
        console.print(Panel(
            performance_text,
            title=f"📊 {self.agent_name} Performance Report",
            border_style="blue"
        ))
    
    def clear_metrics(self):
        """Clear execution metrics."""
        self.execution_metrics.clear()
        self.logger.info(f"Cleared metrics for {self.agent_name}")
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return self.ollama_client.monitor_resources()
    
    def optimize_for_current_gpu(self) -> Dict[str, Any]:
        """Get optimized parameters for current GPU."""
        return self.ollama_client.optimize_for_gpu(self.model) 