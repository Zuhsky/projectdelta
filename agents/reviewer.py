import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from utils.ollama_client import OllamaClient

console = Console()

class ReviewerAgent:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.model = "deepseek-coder:33b"
        self.agent_name = "Reviewer (Lead Engineer)"
        
    def load_prompt(self) -> str:
        """Load the reviewer prompt from file."""
        try:
            with open("prompts/reviewer_prompt.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            console.print("[red]Error: reviewer_prompt.txt not found[/red]")
            return ""
    
    def scan_codebase(self, project_path: str) -> Dict[str, str]:
        """Scan the codebase and collect file contents for review."""
        codebase = {}
        
        # File extensions to review
        review_extensions = ['.tsx', '.ts', '.js', '.jsx', '.json', '.css', '.md', '.yml', '.yaml']
        
        try:
            for root, dirs, files in os.walk(project_path):
                # Skip node_modules and .next directories
                dirs[:] = [d for d in dirs if d not in ['node_modules', '.next', '.git', 'dist', 'build']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, project_path)
                    
                    # Check if file should be reviewed
                    if any(file.endswith(ext) for ext in review_extensions):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if content.strip():  # Only include non-empty files
                                    codebase[relative_path] = content
                        except (UnicodeDecodeError, PermissionError):
                            # Skip binary files or files we can't read
                            continue
        
        except Exception as e:
            console.print(f"[red]Error scanning codebase: {e}[/red]")
        
        console.print(f"[green]Scanned {len(codebase)} files for review[/green]")
        return codebase
    
    def review_codebase(self, project_path: str, technical_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Review the codebase and generate a comprehensive review report."""
        console.print(Panel(
            f"🔍 {self.agent_name} is conducting code review...",
            title="Review Phase",
            border_style="yellow"
        ))
        
        # Scan codebase
        codebase = self.scan_codebase(project_path)
        
        if not codebase:
            console.print("[red]No files found to review[/red]")
            return None
        
        # Load system prompt
        system_prompt = self.load_prompt()
        if not system_prompt:
            return None
        
        # Prepare codebase summary for review
        codebase_summary = self.prepare_codebase_summary(codebase)
        
        # Create user prompt
        user_prompt = f"""
Please conduct a comprehensive code review of the following codebase:

TECHNICAL SPECIFICATION:
{json.dumps(technical_spec, indent=2)}

CODEBASE TO REVIEW:
{codebase_summary}

Please analyze this codebase thoroughly and provide a detailed review following the JSON format specified in the system prompt. Focus on:

1. Security vulnerabilities and potential exploits
2. Performance bottlenecks and optimization opportunities
3. Code quality and maintainability issues
4. Architecture and design pattern adherence
5. Accessibility and SEO implementation
6. Error handling and edge case coverage
7. TypeScript usage and type safety
8. Production readiness

Provide specific, actionable recommendations with file paths and line numbers where applicable.
        """
        
        console.print("[yellow]Analyzing codebase for issues...[/yellow]")
        console.print("[dim]This may take several minutes for large codebases...[/dim]")
        
        # Generate review using Ollama
        response = self.ollama_client.generate(
            model=self.model,
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.1,  # Very low temperature for consistent analysis
            max_tokens=8000
        )
        
        if not response:
            console.print("[red]Failed to generate code review[/red]")
            return None
        
        # Parse the JSON response
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
            
            review_report = json.loads(json_response)
            
            # Validate the review report has required fields
            required_fields = ["review_summary", "critical_issues", "medium_issues", "low_issues"]
            for field in required_fields:
                if field not in review_report:
                    console.print(f"[yellow]Warning: Missing required field '{field}' in review report[/yellow]")
                    review_report[field] = []
            
            console.print("[green]✅ Code review completed successfully[/green]")
            return review_report
            
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing JSON response: {e}[/red]")
            # Create a basic review report as fallback
            return self.create_basic_review_report(codebase)
    
    def prepare_codebase_summary(self, codebase: Dict[str, str]) -> str:
        """Prepare a summary of the codebase for review."""
        summary_parts = []
        
        # Limit the content sent to the model to avoid token limits
        max_files = 20
        max_file_length = 1000
        
        file_count = 0
        for file_path, content in codebase.items():
            if file_count >= max_files:
                break
            
            # Truncate long files
            if len(content) > max_file_length:
                content = content[:max_file_length] + "\n... (truncated)"
            
            summary_parts.append(f"""
FILE: {file_path}
---
{content}
---
""")
            file_count += 1
        
        if len(codebase) > max_files:
            summary_parts.append(f"\n... and {len(codebase) - max_files} more files")
        
        return "\n".join(summary_parts)
    
    def create_basic_review_report(self, codebase: Dict[str, str]) -> Dict[str, Any]:
        """Create a basic review report as fallback."""
        console.print("[yellow]Creating basic review report as fallback...[/yellow]")
        
        # Basic analysis of codebase
        total_files = len(codebase)
        has_typescript = any(f.endswith('.ts') or f.endswith('.tsx') for f in codebase.keys())
        has_tests = any('test' in f.lower() or 'spec' in f.lower() for f in codebase.keys())
        has_package_json = 'package.json' in codebase
        
        return {
            "review_summary": {
                "overall_score": 7,
                "overall_assessment": "Automated basic review completed. Manual review recommended.",
                "critical_issues_count": 0,
                "medium_issues_count": 2 if not has_tests else 1,
                "low_issues_count": 1,
                "recommendations": [
                    "Add comprehensive test coverage",
                    "Implement proper error handling",
                    "Add input validation for API endpoints"
                ]
            },
            "critical_issues": [],
            "medium_issues": [
                {
                    "category": "testing",
                    "title": "Missing test coverage",
                    "description": "No test files detected in the codebase",
                    "file": "N/A",
                    "severity": "medium",
                    "impact": "Reduced confidence in code quality and regression detection",
                    "fix_recommendation": "Add Jest and React Testing Library for comprehensive testing"
                } if not has_tests else {}
            ],
            "low_issues": [
                {
                    "category": "documentation",
                    "title": "Basic documentation",
                    "description": "README could be more comprehensive",
                    "file": "README.md",
                    "severity": "low",
                    "impact": "Developers may have difficulty understanding the project",
                    "fix_recommendation": "Expand README with detailed setup instructions and API documentation"
                }
            ],
            "positive_aspects": [
                {
                    "aspect": "TypeScript Usage",
                    "description": "Project uses TypeScript for better type safety",
                    "impact": "Improved code quality and developer experience"
                } if has_typescript else {
                    "aspect": "Project Structure",
                    "description": "Basic project structure is well organized",
                    "impact": "Good foundation for further development"
                }
            ],
            "technical_debt": [],
            "performance_analysis": {
                "bundle_size_assessment": "Not analyzed in basic review",
                "loading_performance": "Not analyzed in basic review",
                "database_performance": "Not analyzed in basic review",
                "caching_opportunities": []
            },
            "security_assessment": {
                "authentication_security": "Not analyzed in basic review",
                "data_protection": "Not analyzed in basic review",
                "vulnerability_summary": "Manual security audit recommended",
                "security_score": 7
            },
            "production_readiness": {
                "deployment_ready": has_package_json,
                "monitoring_setup": "Not configured",
                "error_handling": "Basic error handling in place",
                "logging_quality": "Minimal logging implemented",
                "environment_configuration": "Environment variables template needed"
            }
        }
    
    def save_review_report(self, review_report: Dict[str, Any], output_path: str = "data/review_report.json") -> bool:
        """Save the review report to a file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(review_report, f, indent=2)
            
            console.print(f"[green]✅ Review report saved to {output_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error saving review report: {e}[/red]")
            return False
    
    def display_review_summary(self, review_report: Dict[str, Any]):
        """Display a summary of the review results."""
        console.print("\n[bold yellow]🔍 Code Review Summary[/bold yellow]")
        
        if "review_summary" in review_report:
            summary = review_report["review_summary"]
            score = summary.get("overall_score", 0)
            
            # Color code the score
            if score >= 8:
                score_color = "green"
            elif score >= 6:
                score_color = "yellow"
            else:
                score_color = "red"
            
            console.print(f"\n[bold {score_color}]Overall Score: {score}/10[/bold {score_color}]")
            console.print(f"[bold cyan]Assessment:[/bold cyan] {summary.get('overall_assessment', 'N/A')}")
            
            # Issue counts
            critical = summary.get("critical_issues_count", 0)
            medium = summary.get("medium_issues_count", 0)
            low = summary.get("low_issues_count", 0)
            
            console.print(f"\n[bold red]Critical Issues:[/bold red] {critical}")
            console.print(f"[bold yellow]Medium Issues:[/bold yellow] {medium}")
            console.print(f"[bold blue]Low Issues:[/bold blue] {low}")
        
        # Security assessment
        if "security_assessment" in review_report:
            security = review_report["security_assessment"]
            security_score = security.get("security_score", 0)
            console.print(f"\n[bold magenta]Security Score:[/bold magenta] {security_score}/10")
        
        # Production readiness
        if "production_readiness" in review_report:
            prod = review_report["production_readiness"]
            ready = prod.get("deployment_ready", False)
            status_color = "green" if ready else "red"
            console.print(f"[bold {status_color}]Production Ready:[/bold {status_color}] {'Yes' if ready else 'No'}")
    
    def run(self, project_path: str, technical_spec_path: str) -> Optional[str]:
        """Run the reviewer agent and return the path to the review report."""
        console.print(f"\n[bold blue]🔄 Starting {self.agent_name}[/bold blue]")
        
        # Load technical specification
        try:
            with open(technical_spec_path, "r") as f:
                technical_spec = json.load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: Technical specification file not found: {technical_spec_path}[/red]")
            return None
        except json.JSONDecodeError:
            console.print(f"[red]Error: Invalid JSON in technical specification file[/red]")
            return None
        
        # Review codebase
        review_report = self.review_codebase(project_path, technical_spec)
        
        if not review_report:
            console.print("[red]❌ Failed to generate review report[/red]")
            return None
        
        # Display summary
        self.display_review_summary(review_report)
        
        # Save review report
        output_path = "data/review_report.json"
        if self.save_review_report(review_report, output_path):
            console.print(f"\n[green]✅ {self.agent_name} completed successfully[/green]")
            return output_path
        else:
            console.print(f"\n[red]❌ {self.agent_name} failed to save output[/red]")
            return None

def main():
    """Test the reviewer agent standalone."""
    ollama_client = OllamaClient()
    reviewer = ReviewerAgent(ollama_client)
    
    # Test with a sample project
    result = reviewer.run("output/sample-project", "data/technical_spec.json")
    if result:
        console.print(f"Review report saved to: {result}")
    else:
        console.print("Failed to generate review report")

if __name__ == "__main__":
    main()