#!/usr/bin/env python3
"""
AI Development Team Orchestrator - GPU Optimized

A comprehensive Python CLI tool that simulates a full-stack software development team
of 6 AI agents, each representing a real professional role in a web development company.
This system generates complex, scalable, production-grade websites using local Ollama models.

Optimized for RunPod GPU environment with 30B+ models.

Author: AI Development Team Orchestrator
Version: 2.0.0
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cli_wizard import CLIWizard
from src.core.orchestrator import GPUOrchestrator
from src.utils.logger import setup_logging

console = Console()

class Orchestrator:
    def __init__(self):
        # Setup logging
        self.logger = setup_logging()
        
        # Initialize GPU-optimized orchestrator
        self.orchestrator = GPUOrchestrator()
        
    def display_welcome(self):
        """Display the welcome banner and team introduction."""
        # Use the GPU-optimized welcome display
        self.orchestrator.display_welcome()
    
    async def check_prerequisites(self) -> bool:
        """Check if Ollama is running and required models are available."""
        return await self.orchestrator.check_gpu_prerequisites()
    
    async def run_development_pipeline(self, project_spec_path: str) -> Optional[str]:
        """Run the complete development pipeline with all agents."""
        return await self.orchestrator.run_development_pipeline_async(project_spec_path)
    
    def display_completion_summary(self, project_path: str):
        """Display the completion summary and next steps."""
        console.print("\n[bold green]🎉 GPU-Optimized Development Pipeline Completed![/bold green]")
        
        # Project summary
        summary_panel = f"""
[bold]Project Location:[/bold] {project_path}

[bold]What was created:[/bold]
• Complete Next.js 14 application with TypeScript
• Responsive UI with Tailwind CSS  
• Database schema with Prisma
• API routes and authentication
• Comprehensive documentation
• Deployment configurations
• Git repository with organized commits

[bold]Quality Assurance:[/bold]
• Code review completed
• Security vulnerabilities addressed  
• Performance optimizations applied
• Accessibility compliance checked
• SEO optimization implemented

[bold]GPU Performance:[/bold]
• Optimized for 30B+ models
• Async processing with GPU acceleration
• Comprehensive performance monitoring
• Resource usage tracking

[bold]Ready for:[/bold]
• Local development (npm run dev)
• Production deployment (Vercel/Netlify)
• Team collaboration (Git workflow)
• Future enhancements
        """
        
        console.print(Panel(summary_panel, title="📊 Project Summary", border_style="green"))
        
        # Display GPU performance summary
        self.orchestrator.display_gpu_performance_summary()
        
        # Next steps
        next_steps = """
[bold yellow]🚀 Next Steps:[/bold yellow]

1. [bold]Review the generated code:[/bold]
   cd {project_path}
   
2. [bold]Install dependencies:[/bold]
   npm install
   
3. [bold]Set up environment variables:[/bold]
   cp .env.example .env.local
   # Edit .env.local with your values
   
4. [bold]Start development server:[/bold]
   npm run dev
   
5. [bold]Deploy to production:[/bold]
   • Push to GitHub
   • Connect to Vercel/Netlify
   • Configure environment variables
   • Deploy!

6. [bold]Read the documentation:[/bold]
   • README.md - Setup and overview
   • docs/API.md - API documentation  
   • docs/DEPLOYMENT.md - Deployment guide
   • docs/USER_GUIDE.md - User manual
        """.format(project_path=project_path)
        
        console.print(next_steps)
        
        # Success message
        console.print("\n[bold green]✨ Your production-ready web application is complete![/bold green]")
        console.print("[green]Built by your GPU-accelerated AI Development Team with enterprise-grade quality standards.[/green]")

async def main():
    """Main entry point for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="AI Development Team Orchestrator - Build production-ready web apps with GPU-accelerated AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run interactive wizard
  python main.py --skip-wizard     # Skip wizard, use existing project_spec.json
  python main.py --help            # Show this help message

For more information, visit: https://github.com/ai-dev-team/orchestrator
        """
    )
    
    parser.add_argument(
        "--skip-wizard",
        action="store_true",
        help="Skip the interactive wizard and use existing data/project_spec.json"
    )
    
    parser.add_argument(
        "--spec-file",
        type=str,
        default="data/project_spec.json",
        help="Path to project specification file (default: data/project_spec.json)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="AI Development Team Orchestrator v2.0.0 (GPU Optimized)"
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Display welcome
    orchestrator.display_welcome()
    
    # Check prerequisites
    if not await orchestrator.check_prerequisites():
        sys.exit(1)
    
    try:
        # Get project specification
        if args.skip_wizard:
            if not os.path.exists(args.spec_file):
                console.print(f"[red]Error: Specification file not found: {args.spec_file}[/red]")
                console.print("[yellow]Run without --skip-wizard to create a new specification[/yellow]")
                sys.exit(1)
            project_spec_path = args.spec_file
            console.print(f"[green]Using existing specification: {project_spec_path}[/green]")
        else:
            # Run interactive wizard
            wizard = CLIWizard()
            project_spec_path = wizard.run_wizard()
            
            if not project_spec_path:
                console.print("[yellow]Wizard cancelled or failed. Exiting.[/yellow]")
                sys.exit(1)
        
        # Run development pipeline
        project_path = await orchestrator.run_development_pipeline(project_spec_path)
        
        if project_path:
            orchestrator.display_completion_summary(project_path)
        else:
            console.print("\n[red]❌ Development pipeline failed[/red]")
            console.print("[yellow]Check the logs above for error details[/yellow]")
            sys.exit(1)
    
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Operation cancelled by user. Goodbye![/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)

def run_main():
    """Run the async main function."""
    asyncio.run(main())

if __name__ == "__main__":
    run_main()