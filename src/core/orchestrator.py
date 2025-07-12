import asyncio
import time
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.live import Live

from ..utils.ollama_client import RunPodOllamaClient
from ..utils.logger import get_logger, setup_logging, log_gpu_status, metrics_logger
from ..core.config import config

console = Console()
logger = get_logger("orchestrator")

class GPUOrchestrator:
    """GPU-optimized orchestrator for RunPod environment."""
    
    def __init__(self):
        self.ollama_client = RunPodOllamaClient()
        self.required_models = config.default_models
        self.agents = {}
        self.performance_metrics = []
        self.logger = logger
        
        # Setup environment
        config.setup_environment()
        
    async def initialize_agents(self):
        """Initialize all agents with GPU optimization."""
        try:
            # Import agents dynamically to avoid circular imports
            from ..agents.planner import PlannerAgent
            from ..agents.builder import BuilderAgent
            from ..agents.reviewer import ReviewerAgent
            from ..agents.fixer import FixerAgent
            from ..agents.finalizer import FinalizerAgent
            from ..agents.git_pusher import GitPusherAgent
            
            # Initialize agents with optimized models
            self.agents = {
                "planner": PlannerAgent(self.ollama_client, "mixtral:8x7b-instruct", "Planner"),
                "builder": BuilderAgent(self.ollama_client, "codellama:70b-instruct", "Builder"),
                "reviewer": ReviewerAgent(self.ollama_client, "deepseek-coder:33b", "Reviewer"),
                "fixer": FixerAgent(self.ollama_client, "deepseek-coder:33b", "Fixer"),
                "finalizer": FinalizerAgent(self.ollama_client, "llama2:70b-chat", "Finalizer"),
                "git_pusher": GitPusherAgent(self.ollama_client, "llama2:70b-chat", "GitPusher")
            }
            
            self.logger.info("All agents initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to import agents: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def display_welcome(self):
        """Display the welcome banner with GPU information."""
        # Get GPU info
        gpu_info = self.ollama_client.get_gpu_info()
        
        banner = f"""
🤖 AI Development Team Orchestrator v2.0.0 - GPU Optimized

Simulating a complete software development team to build production-ready web applications.
Optimized for RunPod GPU environment with 30B+ models.

🖥️ GPU: {gpu_info.get('name', 'Unknown')} | Memory: {gpu_info.get('memory_total', 0)}MB
⚡ Models: {', '.join(self.required_models)}
🚀 Performance: Async processing with GPU acceleration
        """
        
        console.print(Panel(banner, title="🎯 Welcome to GPU-Optimized Development", border_style="bold blue"))
        
        # Display team members with model information
        team_table = Table(title="👥 Your AI Development Team (GPU Optimized)")
        team_table.add_column("Role", style="bold cyan")
        team_table.add_column("Agent", style="green")
        team_table.add_column("Model", style="yellow")
        team_table.add_column("GPU Optimization", style="magenta")
        
        team_members = [
            ("Product Manager", "Planner", "Mixtral 8x7B", "High-context planning"),
            ("Full-Stack Developer", "Builder", "CodeLlama 70B", "Advanced code generation"),
            ("Lead Engineer", "Reviewer", "DeepSeek 33B", "Code quality analysis"),
            ("Senior Debugger", "Fixer", "DeepSeek 33B", "Bug fixing & optimization"),
            ("QA Engineer", "Finalizer", "Llama2 70B", "Documentation & testing"),
            ("DevOps Engineer", "Git Pusher", "Llama2 70B", "Deployment & Git")
        ]
        
        for role, agent, model, optimization in team_members:
            team_table.add_row(role, agent, model, optimization)
        
        console.print(team_table)
        console.print()
    
    async def check_gpu_prerequisites(self) -> bool:
        """Check GPU and model prerequisites with detailed monitoring."""
        console.print("[bold yellow]🔍 Checking GPU prerequisites...[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            
            # Check GPU
            gpu_task = progress.add_task("Checking GPU availability...", total=100)
            gpu_info = self.ollama_client.get_gpu_info()
            
            if gpu_info:
                progress.update(gpu_task, completed=100, description=f"GPU: {gpu_info['name']} ({gpu_info['memory_total']}MB)")
                console.print(f"[green]✅ GPU: {gpu_info['name']} with {gpu_info['memory_total']}MB VRAM[/green]")
                log_gpu_status(self.logger, gpu_info)
            else:
                progress.update(gpu_task, completed=0, description="No GPU detected")
                console.print("[red]❌ No GPU detected[/red]")
                return False
            
            # Check Ollama service
            ollama_task = progress.add_task("Checking Ollama service...", total=100)
            try:
                async with self.ollama_client as client:
                    if not await client.is_model_available("llama2:70b-chat"):
                        progress.update(ollama_task, completed=0, description="Ollama service not accessible")
                        console.print("[red]❌ Ollama service not accessible[/red]")
                        return False
                    progress.update(ollama_task, completed=100, description="Ollama service running")
            except Exception as e:
                progress.update(ollama_task, completed=0, description=f"Ollama error: {e}")
                console.print(f"[red]❌ Ollama service error: {e}[/red]")
                return False
            
            # Check models
            model_task = progress.add_task("Checking 30B+ models...", total=len(self.required_models) * 100)
            
            for i, model in enumerate(self.required_models):
                try:
                    async with self.ollama_client as client:
                        if not await client.is_model_available(model):
                            progress.update(model_task, description=f"Pulling {model}...")
                            console.print(f"[yellow]⚠️ Model {model} not found, pulling...[/yellow]")
                            
                            if not await client.pull_model(model):
                                progress.update(model_task, completed=0, description=f"Failed to pull {model}")
                                console.print(f"[red]❌ Failed to pull {model}[/red]")
                                return False
                            
                            progress.update(model_task, completed=(i + 1) * 100, description=f"Pulled {model}")
                            console.print(f"[green]✅ Successfully pulled {model}[/green]")
                        else:
                            progress.update(model_task, completed=(i + 1) * 100, description=f"Model {model} available")
                            console.print(f"[green]✅ Model {model} is available[/green]")
                            
                except Exception as e:
                    progress.update(model_task, completed=0, description=f"Error with {model}: {e}")
                    console.print(f"[red]❌ Error with {model}: {e}[/red]")
                    return False
            
            progress.update(model_task, completed=len(self.required_models) * 100, description="All models ready")
        
        console.print("\n[green]✅ All GPU prerequisites are met![/green]\n")
        return True
    
    async def run_development_pipeline_async(self, project_spec_path: str) -> Optional[str]:
        """Run the complete development pipeline with GPU optimization and monitoring."""
        console.print("[bold green]🚀 Starting GPU-Optimized Development Pipeline[/bold green]\n")
        
        pipeline_steps = [
            ("Planner", "planner", "Planning & Analysis", "mixtral:8x7b-instruct"),
            ("Builder", "builder", "Development", "codellama:70b-instruct"),
            ("Reviewer", "reviewer", "Code Review", "deepseek-coder:33b"),
            ("Fixer", "fixer", "Bug Fixes & Optimization", "deepseek-coder:33b"),
            ("Finalizer", "finalizer", "QA & Documentation", "llama2:70b-chat"),
            ("Git Pusher", "git_pusher", "Deployment Setup", "llama2:70b-chat")
        ]
        
        technical_spec_path = None
        project_path = None
        review_report_path = None
        
        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        
        try:
            with progress:
                pipeline_task = progress.add_task("Development Pipeline", total=len(pipeline_steps) * 100)
                
                for i, (role, agent_key, phase, model) in enumerate(pipeline_steps, 1):
                    console.print(f"\n[bold blue]═══ Phase {i}/6: {phase} ═══[/bold blue]")
                    
                    agent = self.agents[agent_key]
                    start_time = time.time()
                    
                    # Update progress
                    progress.update(pipeline_task, description=f"Phase {i}: {role} ({model})")
                    
                    try:
                        # Execute agent with monitoring
                        if agent_key == "planner":
                            technical_spec_path = await agent.execute_with_monitoring(
                                agent.run_async, project_spec_path
                            )
                            if not technical_spec_path:
                                console.print(f"[red]❌ {role} failed - stopping pipeline[/red]")
                                return None
                        
                        elif agent_key == "builder":
                            project_path = await agent.execute_with_monitoring(
                                agent.run_async, technical_spec_path
                            )
                            if not project_path:
                                console.print(f"[red]❌ {role} failed - stopping pipeline[/red]")
                                return None
                        
                        elif agent_key == "reviewer":
                            review_report_path = await agent.execute_with_monitoring(
                                agent.run_async, project_path, technical_spec_path
                            )
                            if not review_report_path:
                                console.print(f"[red]❌ {role} failed - stopping pipeline[/red]")
                                return None
                        
                        else:
                            success = await agent.execute_with_monitoring(
                                agent.run_async, 
                                project_path, 
                                review_report_path if agent_key == "fixer" else None
                            )
                            if not success:
                                console.print(f"[yellow]⚠️ {role} had issues but continuing...[/yellow]")
                        
                        # Update progress
                        execution_time = time.time() - start_time
                        progress.update(pipeline_task, completed=i * 100, description=f"Phase {i} completed in {execution_time:.2f}s")
                        
                        # Display performance metrics
                        console.print(f"[dim]⏱️ {role} completed in {execution_time:.2f}s using {model}[/dim]")
                        
                        # Small delay between phases for better UX
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        console.print(f"[red]❌ {role} failed with error: {e}[/red]")
                        self.logger.error(f"Pipeline phase {i} ({role}) failed: {e}")
                        
                        # Continue with next phase if possible
                        if agent_key in ["fixer", "finalizer", "git_pusher"]:
                            console.print(f"[yellow]⚠️ Continuing pipeline despite {role} failure...[/yellow]")
                            continue
                        else:
                            return project_path if project_path else None
            
            return project_path
            
        except Exception as e:
            console.print(f"\n[red]Pipeline failed with error: {e}[/red]")
            self.logger.error(f"Pipeline error: {e}")
            return project_path if project_path else None
    
    def display_gpu_performance_summary(self):
        """Display comprehensive GPU performance summary."""
        console.print("\n[bold green]📊 GPU Performance Summary[/bold green]")
        
        # Collect all metrics
        all_metrics = []
        for agent in self.agents.values():
            all_metrics.extend(agent.execution_metrics)
        
        if not all_metrics:
            console.print("[yellow]No performance metrics available[/yellow]")
            return
        
        # Calculate totals
        total_time = sum(m["execution_time"] for m in all_metrics)
        successful_runs = sum(1 for m in all_metrics if m.get("success"))
        total_runs = len(all_metrics)
        
        # Get GPU info
        gpu_info = self.ollama_client.get_gpu_info()
        
        # Create performance table
        table = Table(title="GPU Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Execution Time", f"{total_time:.2f}s")
        table.add_row("Successful Runs", f"{successful_runs}/{total_runs}")
        table.add_row("Success Rate", f"{(successful_runs/total_runs)*100:.1f}%")
        table.add_row("Average Time per Agent", f"{total_time/len(self.agents):.2f}s")
        
        if gpu_info:
            table.add_row("GPU Model", gpu_info.get("name", "Unknown"))
            table.add_row("GPU Memory Used", f"{gpu_info.get('memory_used', 0)}/{gpu_info.get('memory_total', 0)}MB")
            table.add_row("GPU Temperature", f"{gpu_info.get('temperature', 0):.1f}°C")
            table.add_row("GPU Load", f"{gpu_info.get('load', 0)*100:.1f}%")
        
        # Add cache statistics if available
        cache_stats = self.ollama_client.cache.get_stats() if self.ollama_client.cache else None
        if cache_stats:
            table.add_row("Cache Hit Rate", f"{cache_stats['hit_rate']*100:.1f}%")
            table.add_row("Cache Size", f"{cache_stats['cache_size']}/{cache_stats['max_size']}")
        
        console.print(table)
        
        # Display individual agent performance
        console.print("\n[bold blue]Individual Agent Performance:[/bold blue]")
        for agent_name, agent in self.agents.items():
            agent.display_performance_summary()
    
    def save_performance_report(self, output_path: str = None):
        """Save detailed performance report to file."""
        if output_path is None:
            output_path = Path(config.log_directory) / f"performance_report_{int(time.time())}.json"
        
        report = {
            "timestamp": time.time(),
            "gpu_info": self.ollama_client.get_gpu_info(),
            "cache_stats": self.ollama_client.cache.get_stats() if self.ollama_client.cache else None,
            "agent_performance": {
                name: agent.get_performance_report() for name, agent in self.agents.items()
            },
            "overall_metrics": {
                "total_executions": sum(len(agent.execution_metrics) for agent in self.agents.values()),
                "total_time": sum(sum(m["execution_time"] for m in agent.execution_metrics) for agent in self.agents.values()),
                "success_rate": sum(sum(1 for m in agent.execution_metrics if m.get("success")) for agent in self.agents.values()) / max(sum(len(agent.execution_metrics) for agent in self.agents.values()), 1)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"[green]✅ Performance report saved to: {output_path}[/green]")
    
    async def run_orchestrator(self, project_spec_path: str) -> Optional[str]:
        """Main orchestrator method that runs the complete pipeline."""
        try:
            # Initialize agents
            await self.initialize_agents()
            
            # Display welcome
            self.display_welcome()
            
            # Check prerequisites
            if not await self.check_gpu_prerequisites():
                return None
            
            # Run development pipeline
            project_path = await self.run_development_pipeline_async(project_spec_path)
            
            if project_path:
                # Display performance summary
                self.display_gpu_performance_summary()
                
                # Save performance report
                self.save_performance_report()
                
                return project_path
            else:
                console.print("\n[red]❌ Development pipeline failed[/red]")
                return None
                
        except Exception as e:
            console.print(f"\n[red]Orchestrator failed with error: {e}[/red]")
            self.logger.error(f"Orchestrator error: {e}")
            return None 