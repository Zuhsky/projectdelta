import requests
import json
import asyncio
import aiohttp
import hashlib
import time
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from rich.console import Console
import psutil
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor
import GPUtil
from ..core.config import config
from .logger import get_logger, log_gpu_status, log_model_loading

console = Console()
logger = get_logger("ollama_client")

@dataclass
class OllamaResponse:
    """Structured response from Ollama."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_time: float

class ModelCache:
    """Cache for model responses to improve performance."""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get_cache_key(self, model: str, prompt: str, **kwargs) -> str:
        """Generate cache key from inputs."""
        content = f"{model}:{prompt}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached response."""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached response."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }

class RunPodOllamaClient:
    """GPU-optimized Ollama client for RunPod environment."""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.ollama_base_url
        self.api_url = f"{self.base_url}/api"
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = ModelCache(max_size=200) if config.enable_model_caching else None
        self.logger = logger
        
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=config.request_timeout)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information for optimization."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                return {
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "temperature": gpu.temperature,
                    "load": gpu.load,
                    "uuid": gpu.uuid
                }
        except Exception as e:
            self.logger.warning(f"Could not get GPU info: {e}")
        return {}
    
    def optimize_for_gpu(self, model: str) -> Dict[str, Any]:
        """Optimize generation parameters for GPU."""
        gpu_info = self.get_gpu_info()
        model_config = config.get_model_config(model)
        
        # Adjust based on available GPU memory
        if gpu_info.get("memory_total", 0) >= 24000:  # 24GB+ GPU
            model_config.update({
                "num_ctx": 32768,
                "num_thread": 8,
                "num_gpu": 1
            })
        elif gpu_info.get("memory_total", 0) >= 12000:  # 12GB+ GPU
            model_config.update({
                "num_ctx": 16384,
                "num_thread": 6,
                "num_gpu": 1
            })
        else:  # Smaller GPU
            model_config.update({
                "num_ctx": 8192,
                "num_thread": 4,
                "num_gpu": 1
            })
        
        return model_config
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def generate_async(
        self, 
        model: str, 
        prompt: str, 
        system: Optional[str] = None,
        **kwargs
    ) -> Optional[OllamaResponse]:
        """Async generation with GPU optimization."""
        if not self.session:
            raise RuntimeError("Client not initialized with async context")
        
        start_time = time.time()
        
        # Check cache first
        if self.cache:
            cache_key = self.cache.get_cache_key(model, prompt, system=system, **kwargs)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                self.logger.info(f"Cache hit for model {model}")
                return cached_response
        
        # Optimize parameters for GPU
        options = self.optimize_for_gpu(model)
        options.update(kwargs)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options
        }
        
        if system:
            payload["system"] = system
        
        try:
            async with self.session.post(
                f"{self.api_url}/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    response_time = time.time() - start_time
                    
                    ollama_response = OllamaResponse(
                        content=result.get("response", ""),
                        model=model,
                        usage=result.get("usage", {}),
                        finish_reason=result.get("done_reason", "unknown"),
                        response_time=response_time
                    )
                    
                    # Cache the response
                    if self.cache:
                        cache_key = self.cache.get_cache_key(model, prompt, system=system, **kwargs)
                        self.cache.set(cache_key, ollama_response)
                    
                    self.logger.info(f"Generated response in {response_time:.2f}s using {model}")
                    return ollama_response
                else:
                    error_text = await response.text()
                    self.logger.error(f"Ollama API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    def generate(
        self, 
        model: str, 
        prompt: str, 
        system: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """Synchronous generation wrapper."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def _generate():
                async with self as client:
                    response = await client.generate_async(model, prompt, system, **kwargs)
                    return response.content if response else None
            
            return loop.run_until_complete(_generate())
        finally:
            loop.close()
    
    async def generate_batch_async(
        self, 
        model: str, 
        prompts: List[str], 
        system: Optional[str] = None,
        max_concurrent: int = 4
    ) -> List[Optional[OllamaResponse]]:
        """Generate responses for multiple prompts in batch with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(prompt: str) -> Optional[OllamaResponse]:
            async with semaphore:
                return await self.generate_async(model, prompt, system)
        
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch generation error: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def generate_batch(
        self, 
        model: str, 
        prompts: List[str], 
        system: Optional[str] = None
    ) -> List[Optional[str]]:
        """Synchronous batch generation wrapper."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def _batch_generate():
                async with self as client:
                    responses = await client.generate_batch_async(model, prompts, system)
                    return [r.content if r else None for r in responses]
            
            return loop.run_until_complete(_batch_generate())
        finally:
            loop.close()
    
    async def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available asynchronously."""
        try:
            async with self.session.get(f"{self.api_url}/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    return any(model["name"].startswith(model_name) for model in models)
                return False
        except Exception as e:
            self.logger.error(f"Error checking model availability: {e}")
            return False
    
    def is_model_available_sync(self, model_name: str) -> bool:
        """Synchronous model availability check."""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"].startswith(model_name) for model in models)
            return False
        except Exception as e:
            self.logger.error(f"Error checking model availability: {e}")
            return False
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from the Ollama registry asynchronously."""
        try:
            self.logger.info(f"Pulling model {model_name}...")
            log_model_loading(self.logger, model_name, "loading")
            
            payload = {"name": model_name}
            
            async with self.session.post(
                f"{self.api_url}/pull",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=1800)  # 30 minute timeout for large models
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if "status" in data:
                                    self.logger.info(f"Model pull status: {data['status']}")
                            except json.JSONDecodeError:
                                continue
                    
                    log_model_loading(self.logger, model_name, "loaded")
                    return True
                else:
                    log_model_loading(self.logger, model_name, "error")
                    return False
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            log_model_loading(self.logger, model_name, "error")
            return False
    
    def pull_model_sync(self, model_name: str) -> bool:
        """Synchronous model pull."""
        try:
            self.logger.info(f"Pulling model {model_name}...")
            log_model_loading(self.logger, model_name, "loading")
            
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": model_name},
                stream=True,
                timeout=1800
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if "status" in data:
                                self.logger.info(f"Model pull status: {data['status']}")
                        except json.JSONDecodeError:
                            continue
                
                log_model_loading(self.logger, model_name, "loaded")
                return True
            else:
                log_model_loading(self.logger, model_name, "error")
                return False
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            log_model_loading(self.logger, model_name, "error")
            return False
    
    async def ensure_models_available(self, models: List[str]) -> bool:
        """Ensure all required models are available, pull if necessary."""
        for model in models:
            if not await self.is_model_available(model):
                self.logger.info(f"Model {model} not found locally. Pulling...")
                if not await self.pull_model(model):
                    self.logger.error(f"Failed to pull model {model}")
                    return False
                self.logger.info(f"Successfully pulled {model}")
            else:
                self.logger.info(f"Model {model} is available")
        return True
    
    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor system resources."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available": psutil.virtual_memory().available // (1024**3),  # GB
            "gpu_info": self.get_gpu_info(),
            "cache_stats": self.cache.get_stats() if self.cache else None
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "cache_stats": self.cache.get_stats() if self.cache else None,
            "resource_usage": self.monitor_resources()
        } 