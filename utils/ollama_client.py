import requests
import json
from typing import Optional, Dict, Any
from rich.console import Console

console = Console()

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available locally."""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"].startswith(model_name) for model in models)
            return False
        except requests.exceptions.RequestException:
            return False
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from the Ollama registry."""
        try:
            console.print(f"[yellow]Pulling model {model_name}...[/yellow]")
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": model_name},
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if "status" in data:
                            console.print(f"[blue]{data['status']}[/blue]")
                return True
            return False
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error pulling model: {e}[/red]")
            return False
    
    def generate(self, model: str, prompt: str, system: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: Optional[int] = None) -> Optional[str]:
        """Generate a response using the specified model."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if system:
                payload["system"] = system
            
            if max_tokens:
                payload["options"]["num_ctx"] = max_tokens
            
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=300  # 5 minute timeout for complex generations
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                console.print(f"[red]Error: {response.status_code} - {response.text}[/red]")
                return None
                
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Request failed: {e}[/red]")
            return None
    
    def chat(self, model: str, messages: list, temperature: float = 0.7) -> Optional[str]:
        """Chat with a model using conversation format."""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                console.print(f"[red]Error: {response.status_code} - {response.text}[/red]")
                return None
                
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Request failed: {e}[/red]")
            return None
    
    def ensure_models_available(self, models: list) -> bool:
        """Ensure all required models are available, pull if necessary."""
        for model in models:
            if not self.is_model_available(model):
                console.print(f"[yellow]Model {model} not found locally. Pulling...[/yellow]")
                if not self.pull_model(model):
                    console.print(f"[red]Failed to pull model {model}[/red]")
                    return False
                console.print(f"[green]Successfully pulled {model}[/green]")
            else:
                console.print(f"[green]Model {model} is available[/green]")
        return True