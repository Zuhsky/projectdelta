from pydantic import BaseSettings, Field
from typing import List, Optional, Dict, Any
import os
from pathlib import Path

class RunPodConfig(BaseSettings):
    """RunPod-optimized configuration for high-performance GPUs."""
    
    # GPU Optimization
    cuda_visible_devices: str = Field(default="0", description="GPU device to use")
    torch_cuda_alloc_conf: str = Field(default="max_split_size_mb:1024", description="PyTorch CUDA memory allocation")
    ollama_max_loaded_models: int = Field(default=3, description="Maximum models to keep in memory")
    ollama_num_parallel: int = Field(default=8, description="Number of parallel processing threads")
    ollama_max_queue: int = Field(default=1024, description="Maximum request queue size")
    
    # Model Configuration for 30B+ models
    default_models: List[str] = Field(
        default=[
            "llama2:70b-chat",           # 70B model for complex reasoning
            "deepseek-coder:33b",        # 33B for code generation
            "codellama:70b-instruct",    # 70B for advanced coding
            "mixtral:8x7b-instruct"      # Mixture of experts for planning
        ],
        description="List of models to use"
    )
    
    # Performance Settings
    model_temperature: float = Field(default=0.3, description="Model temperature for generation")
    max_tokens: int = Field(default=32000, description="Maximum tokens for generation")
    batch_size: int = Field(default=4, description="Batch size for processing")
    request_timeout: int = Field(default=600, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # RunPod Environment
    runpod_workspace: str = Field(default="/workspace", description="RunPod workspace directory")
    output_directory: str = Field(default="/workspace/output", description="Output directory for generated projects")
    data_directory: str = Field(default="/workspace/data", description="Data directory")
    log_directory: str = Field(default="/workspace/logs", description="Log directory")
    
    # Memory Management
    enable_model_caching: bool = Field(default=True, description="Enable model response caching")
    cache_size_gb: int = Field(default=8, description="Cache size in GB")
    enable_gpu_memory_optimization: bool = Field(default=True, description="Enable GPU memory optimization")
    
    # Ollama Settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    ollama_host: str = Field(default="0.0.0.0:11434", description="Ollama host binding")
    ollama_origins: str = Field(default="*", description="Allowed CORS origins")
    
    # Development Settings
    log_level: str = Field(default="INFO", description="Logging level")
    enable_debug: bool = Field(default=False, description="Enable debug mode")
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    
    class Config:
        env_file = ".env"
        env_prefix = "RUNPOD_"
        case_sensitive = False
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get optimized configuration for specific model."""
        base_config = {
            "temperature": self.model_temperature,
            "max_tokens": self.max_tokens,
            "num_ctx": 32768 if "70b" in model_name else 16384,
            "num_gpu": 1,
            "num_thread": 8 if "70b" in model_name else 4,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1
        }
        
        # Model-specific optimizations
        if "llama2:70b" in model_name:
            base_config.update({
                "num_ctx": 32768,
                "num_thread": 8,
                "temperature": 0.2  # Lower for reasoning tasks
            })
        elif "deepseek-coder" in model_name:
            base_config.update({
                "num_ctx": 16384,
                "num_thread": 6,
                "temperature": 0.1  # Very low for code generation
            })
        elif "codellama" in model_name:
            base_config.update({
                "num_ctx": 32768,
                "num_thread": 8,
                "temperature": 0.1  # Very low for code generation
            })
        elif "mixtral" in model_name:
            base_config.update({
                "num_ctx": 16384,
                "num_thread": 6,
                "temperature": 0.3  # Balanced for planning
            })
        
        return base_config
    
    def setup_environment(self):
        """Setup environment variables for GPU optimization."""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.torch_cuda_alloc_conf
        os.environ["OLLAMA_HOST"] = self.ollama_host
        os.environ["OLLAMA_ORIGINS"] = self.ollama_origins
        os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(self.ollama_max_loaded_models)
        os.environ["OLLAMA_NUM_PARALLEL"] = str(self.ollama_num_parallel)
        os.environ["OLLAMA_MAX_QUEUE"] = str(self.ollama_max_queue)
        
        # Create directories
        for directory in [self.output_directory, self.data_directory, self.log_directory]:
            Path(directory).mkdir(parents=True, exist_ok=True)

# Global configuration instance
config = RunPodConfig() 