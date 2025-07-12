import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from ..core.config import config

console = Console()

class GPUOptimizedFormatter(logging.Formatter):
    """Custom formatter for GPU-optimized logging."""
    
    def format(self, record):
        # Add GPU info if available
        if hasattr(record, 'gpu_info'):
            record.gpu_info_str = f"GPU: {record.gpu_info}"
        else:
            record.gpu_info_str = ""
        
        # Add execution time if available
        if hasattr(record, 'execution_time'):
            record.execution_time_str = f"⏱️ {record.execution_time:.2f}s"
        else:
            record.execution_time_str = ""
        
        return super().format(record)

class MetricsLogger:
    """Logger for performance metrics and GPU monitoring."""
    
    def __init__(self, log_file: str = "gpu_metrics.json"):
        self.log_file = Path(config.log_directory) / log_file
        self.metrics = []
    
    def log_agent_execution(self, agent_name: str, execution_time: float, 
                           success: bool, gpu_info: Optional[Dict] = None, 
                           error: Optional[str] = None):
        """Log agent execution metrics."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "execution_time": execution_time,
            "success": success,
            "gpu_info": gpu_info,
            "error": error
        }
        
        self.metrics.append(metric)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from logged metrics."""
        if not self.metrics:
            return {}
        
        successful_runs = [m for m in self.metrics if m["success"]]
        failed_runs = [m for m in self.metrics if not m["success"]]
        
        return {
            "total_executions": len(self.metrics),
            "successful_executions": len(successful_runs),
            "failed_executions": len(failed_runs),
            "success_rate": len(successful_runs) / len(self.metrics) if self.metrics else 0,
            "average_execution_time": sum(m["execution_time"] for m in successful_runs) / len(successful_runs) if successful_runs else 0,
            "total_execution_time": sum(m["execution_time"] for m in self.metrics),
            "fastest_execution": min(m["execution_time"] for m in self.metrics) if self.metrics else 0,
            "slowest_execution": max(m["execution_time"] for m in self.metrics) if self.metrics else 0
        }

def setup_logging(level: str = None, log_file: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging system for GPU environment."""
    
    # Use config level if not specified
    if level is None:
        level = config.log_level
    
    # Create logs directory
    log_dir = Path(config.log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create main logger
    logger = logging.getLogger("ai_orchestrator")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with Rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True
    )
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # File handler for detailed logs
    if log_file is None:
        log_file = log_dir / f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_file = log_dir / log_file
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Custom formatter for file logs
    file_formatter = GPUOptimizedFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s %(gpu_info_str)s %(execution_time_str)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log startup information
    logger.info("🚀 AI Development Team Orchestrator starting up...")
    logger.info(f"📁 Log directory: {log_dir}")
    logger.info(f"🔧 Log level: {level}")
    logger.info(f"🎯 GPU optimization enabled: {config.enable_gpu_memory_optimization}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(f"ai_orchestrator.{name}")

def log_gpu_status(logger: logging.Logger, gpu_info: Dict[str, Any]):
    """Log GPU status information."""
    if gpu_info:
        logger.info(f"🖥️ GPU Status: {gpu_info['name']} | "
                   f"Memory: {gpu_info['memory_used']}/{gpu_info['memory_total']}MB "
                   f"({gpu_info['memory_used']/gpu_info['memory_total']*100:.1f}%) | "
                   f"Temp: {gpu_info['temperature']}°C | "
                   f"Load: {gpu_info['load']*100:.1f}%")
    else:
        logger.warning("⚠️ No GPU information available")

def log_agent_start(logger: logging.Logger, agent_name: str, model: str):
    """Log agent start with model information."""
    logger.info(f"🤖 Starting {agent_name} with model: {model}")

def log_agent_complete(logger: logging.Logger, agent_name: str, execution_time: float, success: bool):
    """Log agent completion with performance metrics."""
    status = "✅" if success else "❌"
    logger.info(f"{status} {agent_name} completed in {execution_time:.2f}s")

def log_model_loading(logger: logging.Logger, model_name: str, status: str):
    """Log model loading status."""
    if status == "loading":
        logger.info(f"📥 Loading model: {model_name}")
    elif status == "loaded":
        logger.info(f"✅ Model loaded: {model_name}")
    elif status == "error":
        logger.error(f"❌ Failed to load model: {model_name}")

# Global metrics logger instance
metrics_logger = MetricsLogger() 