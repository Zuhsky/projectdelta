import json
import os
import re
from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel

from utils.ollama_client import OllamaClient

console = Console()

class PlannerAgent:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.model = "llama2:7b-chat"
        self.agent_name = "Planner (Product Manager)"
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean control characters and fix common JSON issues."""
        # Remove problematic control characters except for standard whitespace
        json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', json_str)
        
        # Try to parse as-is first
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # If parsing fails, try to fix common issues
            # Remove any trailing commas before closing braces/brackets
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fix unescaped newlines and tabs in strings
            json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            
            return json_str
        
    def load_prompt(self) -> str:
        """Load the planner prompt from file."""
        try:
            with open("prompts/planner_prompt.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            console.print("[red]Error: planner_prompt.txt not found[/red]")
            return ""
    
    def analyze_requirements(self, project_spec_path: str) -> Optional[Dict[str, Any]]:
        """Analyze project requirements and create detailed technical specifications."""
        console.print(Panel(
            f"🔍 {self.agent_name} is analyzing project requirements...",
            title="Planning Phase",
            border_style="blue"
        ))
        
        # Load project specification
        try:
            with open(project_spec_path, "r") as f:
                project_spec = json.load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: Project specification file not found: {project_spec_path}[/red]")
            return None
        except json.JSONDecodeError:
            console.print(f"[red]Error: Invalid JSON in project specification file[/red]")
            return None
        
        # Load system prompt
        system_prompt = self.load_prompt()
        if not system_prompt:
            return None
        
        # Create user prompt with project specification
        user_prompt = f"""
Please analyze the following project specification and create a comprehensive technical specification:

PROJECT SPECIFICATION:
{json.dumps(project_spec, indent=2)}

Based on this specification, provide a detailed technical analysis that will guide the development team in building a production-ready web application. Focus on creating actionable specifications that can be directly implemented.

Remember to:
1. Extract detailed business requirements and technical needs
2. Design a scalable and maintainable architecture
3. Specify all necessary components, APIs, and data models
4. Create a realistic development roadmap
5. Consider security, performance, and user experience
6. Provide enterprise-level specifications

Respond with a comprehensive JSON specification following the format specified in the system prompt.
        """
        
        console.print("[yellow]Generating technical specifications...[/yellow]")
        
        # Try multiple times with different settings if JSON parsing fails
        for attempt in range(3):
            temperature = 0.3 + (attempt * 0.1)  # Start at 0.3, then 0.4, 0.5
            
            if attempt > 0:
                console.print(f"[yellow]Retry attempt {attempt + 1}/3 with temperature {temperature}[/yellow]")
            
            # Generate specifications using Ollama
            response = self.ollama_client.generate(
                model=self.model,
                prompt=user_prompt,
                system=system_prompt,
                temperature=temperature
            )
            
            if not response:
                console.print("[red]Failed to generate response from model[/red]")
                continue
            
            # Try to parse the response
            try:
                # Clean up the response to extract JSON
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.endswith("```"):
                    response = response[:-3]
                if response.startswith("```"):
                    response = response[3:]
                
                # Find the JSON object boundaries
                response = response.strip()
                
                # Try to find the start and end of the JSON object
                start_idx = response.find('{')
                if start_idx == -1:
                    raise json.JSONDecodeError("No JSON object found", response, 0)
                
                # Find the matching closing brace
                brace_count = 0
                end_idx = start_idx
                
                for i, char in enumerate(response[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
                
                # Extract just the JSON part
                json_response = response[start_idx:end_idx + 1]
                
                # Clean up control characters that might cause JSON parsing issues
                json_response = self._clean_json_string(json_response)
                
                technical_spec = json.loads(json_response)
                
                # Validate the specification has required fields
                required_fields = ["project_overview", "technical_stack", "features", "data_models"]
                for field in required_fields:
                    if field not in technical_spec:
                        console.print(f"[yellow]Warning: Missing required field '{field}' in technical specification[/yellow]")
                
                console.print("[green]✅ Technical specifications generated successfully[/green]")
                return technical_spec
                
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Attempt {attempt + 1} failed: {e}[/yellow]")
                if attempt == 2:  # Last attempt
                    console.print(f"[red]Error parsing JSON response: {e}[/red]")
                    console.print(f"[red]Response was: {response[:500]}...[/red]")
                    break
                continue
        
        # If all attempts fail, try fallback
        console.print("[yellow]All attempts failed. Creating fallback specification...[/yellow]")
        
        try:
            # Create a minimal valid technical spec as fallback
            # Load the original project spec to extract basic info
            with open(project_spec_path, "r") as f:
                project_spec = json.load(f)
            
            fallback_spec = {
                "project_overview": {
                    "name": project_spec.get("project_name", "Web Application"),
                    "description": project_spec.get("project_description", "A modern web application"),
                    "target_audience": project_spec.get("target_audience", "General users"),
                    "business_goals": ["Provide value to users", "Generate revenue"],
                    "success_metrics": ["User engagement", "Conversion rate"]
                },
                "technical_stack": {
                    "frontend": "Next.js 14 with TypeScript",
                    "backend": "Next.js API Routes",
                    "database": "PostgreSQL with Prisma",
                    "authentication": "NextAuth.js",
                    "deployment": "Vercel",
                    "third_party_services": []
                },
                "features": [
                    {
                        "name": "User Authentication",
                        "description": "Secure user login and registration",
                        "priority": "high",
                        "user_stories": ["As a user, I want to create an account"],
                        "acceptance_criteria": ["User can register and login"],
                        "estimated_complexity": "medium"
                    }
                ],
                "data_models": [
                    {
                        "name": "User",
                        "fields": [
                            {"name": "id", "type": "string", "required": True, "description": "Unique identifier"},
                            {"name": "email", "type": "string", "required": True, "description": "User email"},
                            {"name": "name", "type": "string", "required": True, "description": "User name"}
                        ],
                        "relationships": []
                    }
                ],
                "api_endpoints": [
                    {
                        "method": "GET",
                        "path": "/api/auth/session",
                        "description": "Get current user session",
                        "parameters": [],
                        "response": "User session data",
                        "authentication_required": True
                    }
                ],
                "pages_and_components": [
                    {
                        "page": "Home",
                        "url": "/",
                        "description": "Landing page",
                        "components": ["Header", "Hero", "Footer"],
                        "authentication_required": False
                    }
                ],
                "file_structure": {
                    "description": "Standard Next.js 14 project structure",
                    "structure": "app/, components/, lib/, public/, styles/"
                },
                "development_phases": [
                    {
                        "phase": "Phase 1: Foundation",
                        "description": "Set up basic structure and authentication",
                        "features": ["Authentication", "Basic UI"],
                        "estimated_time": "1-2 weeks"
                    }
                ],
                "non_functional_requirements": {
                    "performance": "Fast loading times, optimized for web vitals",
                    "security": "Secure authentication, input validation",
                    "scalability": "Horizontal scaling with cloud deployment",
                    "accessibility": "WCAG 2.1 AA compliance",
                    "seo": "SEO optimized with Next.js features"
                }
            }
            
            console.print("[yellow]✅ Using fallback technical specification[/yellow]")
            return fallback_spec
            
        except Exception as fallback_error:
            console.print(f"[red]Fallback specification creation failed: {fallback_error}[/red]")
            return None
    
    def save_technical_spec(self, technical_spec: Dict[str, Any], output_path: str = "data/technical_spec.json") -> bool:
        """Save the technical specification to a file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(technical_spec, f, indent=2)
            
            console.print(f"[green]✅ Technical specification saved to {output_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error saving technical specification: {e}[/red]")
            return False
    
    def display_summary(self, technical_spec: Dict[str, Any]):
        """Display a summary of the technical specifications."""
        console.print("\n[bold green]📋 Technical Specification Summary[/bold green]")
        
        # Project Overview
        if "project_overview" in technical_spec:
            overview = technical_spec["project_overview"]
            console.print(f"\n[bold cyan]Project:[/bold cyan] {overview.get('name', 'N/A')}")
            console.print(f"[bold cyan]Description:[/bold cyan] {overview.get('description', 'N/A')}")
            console.print(f"[bold cyan]Target Audience:[/bold cyan] {overview.get('target_audience', 'N/A')}")
        
        # Technical Stack
        if "technical_stack" in technical_spec:
            stack = technical_spec["technical_stack"]
            console.print(f"\n[bold yellow]Tech Stack:[/bold yellow]")
            for key, value in stack.items():
                console.print(f"  • {key.title()}: {value}")
        
        # Features Count
        if "features" in technical_spec:
            features_count = len(technical_spec["features"])
            high_priority = len([f for f in technical_spec["features"] if f.get("priority") == "high"])
            console.print(f"\n[bold magenta]Features:[/bold magenta] {features_count} total ({high_priority} high priority)")
        
        # Data Models
        if "data_models" in technical_spec:
            models_count = len(technical_spec["data_models"])
            console.print(f"[bold magenta]Data Models:[/bold magenta] {models_count} models defined")
        
        # API Endpoints
        if "api_endpoints" in technical_spec:
            endpoints_count = len(technical_spec["api_endpoints"])
            console.print(f"[bold magenta]API Endpoints:[/bold magenta] {endpoints_count} endpoints specified")
        
        # Development Phases
        if "development_phases" in technical_spec:
            phases_count = len(technical_spec["development_phases"])
            console.print(f"[bold magenta]Development Phases:[/bold magenta] {phases_count} phases planned")
    
    def run(self, project_spec_path: str) -> Optional[str]:
        """Run the planner agent and return the path to the technical specification."""
        console.print(f"\n[bold blue]🔄 Starting {self.agent_name}[/bold blue]")
        
        # Analyze requirements and generate technical specifications
        technical_spec = self.analyze_requirements(project_spec_path)
        
        if not technical_spec:
            console.print("[red]❌ Failed to generate technical specifications[/red]")
            return None
        
        # Display summary
        self.display_summary(technical_spec)
        
        # Save technical specification
        output_path = "data/technical_spec.json"
        if self.save_technical_spec(technical_spec, output_path):
            console.print(f"\n[green]✅ {self.agent_name} completed successfully[/green]")
            return output_path
        else:
            console.print(f"\n[red]❌ {self.agent_name} failed to save output[/red]")
            return None

def main():
    """Test the planner agent standalone."""
    ollama_client = OllamaClient()
    planner = PlannerAgent(ollama_client)
    
    # Test with a sample project spec
    result = planner.run("data/project_spec.json")
    if result:
        console.print(f"Technical specification saved to: {result}")
    else:
        console.print("Failed to generate technical specification")

if __name__ == "__main__":
    main()