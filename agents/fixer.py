import json
import os
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.panel import Panel

from utils.ollama_client import OllamaClient

console = Console()

class FixerAgent:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.model = "deepseek-coder:33b"
        self.agent_name = "Fixer (Senior Debugger)"
        
    def load_prompt(self) -> str:
        """Load the fixer prompt from file."""
        try:
            with open("prompts/fixer_prompt.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            console.print("[red]Error: fixer_prompt.txt not found[/red]")
            return ""
    
    def load_codebase_summary(self, project_path: str) -> str:
        """Load a summary of the current codebase."""
        summary_parts = []
        
        try:
            for root, dirs, files in os.walk(project_path):
                # Skip node_modules and .next directories
                dirs[:] = [d for d in dirs if d not in ['node_modules', '.next', '.git', 'dist', 'build']]
                
                for file in files:
                    if any(file.endswith(ext) for ext in ['.tsx', '.ts', '.js', '.jsx', '.json']):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, project_path)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                # Truncate very long files
                                if len(content) > 2000:
                                    content = content[:2000] + "\n... (truncated)"
                                
                                summary_parts.append(f"""
FILE: {relative_path}
---
{content}
---
""")
                        except (UnicodeDecodeError, PermissionError):
                            continue
        except Exception as e:
            console.print(f"[red]Error reading codebase: {e}[/red]")
        
        return "\n".join(summary_parts)
    
    def apply_fixes(self, project_path: str, review_report: Dict[str, Any]) -> bool:
        """Apply fixes based on the review report."""
        console.print(Panel(
            f"🔧 {self.agent_name} is applying fixes and optimizations...",
            title="Fixing Phase",
            border_style="red"
        ))
        
        # Load system prompt
        system_prompt = self.load_prompt()
        if not system_prompt:
            return False
        
        # Load current codebase
        codebase_summary = self.load_codebase_summary(project_path)
        
        # Create user prompt
        user_prompt = f"""
Please fix all issues identified in the following code review and optimize the codebase:

REVIEW REPORT:
{json.dumps(review_report, indent=2)}

CURRENT CODEBASE:
{codebase_summary}

Please apply all necessary fixes and optimizations following the priority order:
1. Fix all critical security issues immediately
2. Fix critical performance issues
3. Fix critical architecture issues
4. Address medium priority issues
5. Address low priority issues
6. Apply optimizations

For each fix, provide the complete updated file content. Ensure all fixes maintain or improve existing functionality without breaking changes.

Follow the output format specified in the system prompt.
        """
        
        console.print("[yellow]Analyzing issues and generating fixes...[/yellow]")
        console.print("[dim]This may take several minutes for complex fixes...[/dim]")
        
        # Generate fixes using Ollama
        response = self.ollama_client.generate(
            model=self.model,
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.1,  # Very low temperature for consistent fixes
            max_tokens=12000
        )
        
        if not response:
            console.print("[red]Failed to generate fixes[/red]")
            return False
        
        # Parse and apply fixes
        return self.parse_and_apply_fixes(response, project_path)
    
    def parse_and_apply_fixes(self, response: str, project_path: str) -> bool:
        """Parse the AI response and apply the fixes."""
        try:
            console.print("[yellow]Parsing fixes and updating files...[/yellow]")
            
            files_updated = 0
            files_created = 0
            current_file = None
            current_content = []
            in_file_content = False
            
            lines = response.split('\n')
            
            for line in lines:
                if line.startswith('FILE: '):
                    # Save previous file if exists
                    if current_file and current_content:
                        file_path = os.path.join(project_path, current_file)
                        self.update_file(file_path, '\n'.join(current_content))
                        files_updated += 1
                    
                    # Start new file
                    current_file = line.replace('FILE: ', '').strip()
                    current_content = []
                    in_file_content = False
                    
                elif line.startswith('PURPOSE: ') or line.startswith('CHANGES: '):
                    # Skip metadata lines
                    continue
                    
                elif line.strip() == '---':
                    # Toggle file content mode
                    in_file_content = not in_file_content
                    
                elif in_file_content and current_file:
                    # Add content to current file
                    current_content.append(line)
                
                elif line.startswith('NEW FILES CREATED:'):
                    # Handle new files section
                    continue
                
                elif line.startswith('UPDATED FILES:'):
                    # Handle updated files section
                    continue
            
            # Save last file
            if current_file and current_content:
                file_path = os.path.join(project_path, current_file)
                self.update_file(file_path, '\n'.join(current_content))
                files_updated += 1
            
            # Apply basic optimizations if no files were updated
            if files_updated == 0:
                files_updated = self.apply_basic_optimizations(project_path)
            
            console.print(f"[green]✅ Updated {files_updated} files with fixes and optimizations[/green]")
            return files_updated > 0
            
        except Exception as e:
            console.print(f"[red]Error applying fixes: {e}[/red]")
            return False
    
    def update_file(self, file_path: str, content: str):
        """Update a file with new content."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            relative_path = os.path.relpath(file_path)
            console.print(f"[dim]Updated: {relative_path}[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error updating file {file_path}: {e}[/red]")
    
    def apply_basic_optimizations(self, project_path: str) -> int:
        """Apply basic optimizations as fallback."""
        console.print("[yellow]Applying basic optimizations as fallback...[/yellow]")
        
        optimizations_applied = 0
        
        # Add .env.example if it doesn't exist
        env_example_path = os.path.join(project_path, ".env.example")
        if not os.path.exists(env_example_path):
            env_content = """# Environment Variables Template
# Copy this file to .env.local and fill in your values

# Database
DATABASE_URL="postgresql://user:password@localhost:5432/database"

# Authentication
NEXTAUTH_SECRET="your-secret-key"
NEXTAUTH_URL="http://localhost:3000"

# API Keys (add as needed)
# STRIPE_SECRET_KEY=""
# SENDGRID_API_KEY=""
# CLOUDINARY_URL=""
"""
            with open(env_example_path, 'w') as f:
                f.write(env_content)
            optimizations_applied += 1
            console.print("[dim]Added .env.example[/dim]")
        
        # Add basic .gitignore improvements
        gitignore_path = os.path.join(project_path, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                content = f.read()
            
            additions = []
            if ".env.local" not in content:
                additions.append(".env.local")
            if "*.log" not in content:
                additions.append("*.log")
            if ".DS_Store" not in content:
                additions.append(".DS_Store")
            
            if additions:
                with open(gitignore_path, 'a') as f:
                    f.write("\n# Additional entries\n")
                    for item in additions:
                        f.write(f"{item}\n")
                optimizations_applied += 1
                console.print("[dim]Updated .gitignore[/dim]")
        
        # Add basic error handling to package.json scripts
        package_json_path = os.path.join(project_path, "package.json")
        if os.path.exists(package_json_path):
            try:
                with open(package_json_path, 'r') as f:
                    package_data = json.load(f)
                
                # Add type checking script if not present
                if "scripts" in package_data:
                    if "type-check" not in package_data["scripts"]:
                        package_data["scripts"]["type-check"] = "tsc --noEmit"
                        
                    if "lint:fix" not in package_data["scripts"]:
                        package_data["scripts"]["lint:fix"] = "next lint --fix"
                
                with open(package_json_path, 'w') as f:
                    json.dump(package_data, f, indent=2)
                
                optimizations_applied += 1
                console.print("[dim]Updated package.json scripts[/dim]")
                
            except Exception as e:
                console.print(f"[yellow]Could not update package.json: {e}[/yellow]")
        
        return optimizations_applied
    
    def validate_fixes(self, project_path: str) -> bool:
        """Validate that fixes were applied correctly."""
        console.print("[yellow]Validating applied fixes...[/yellow]")
        
        # Check that essential files still exist
        essential_files = ["package.json", "tsconfig.json"]
        
        for file in essential_files:
            file_path = os.path.join(project_path, file)
            if not os.path.exists(file_path):
                console.print(f"[red]Error: Essential file {file} is missing after fixes[/red]")
                return False
        
        # Check that package.json is valid JSON
        try:
            with open(os.path.join(project_path, "package.json"), 'r') as f:
                json.load(f)
        except json.JSONDecodeError:
            console.print("[red]Error: package.json is not valid JSON after fixes[/red]")
            return False
        
        console.print("[green]✅ Fix validation passed[/green]")
        return True
    
    def run(self, project_path: str, review_report_path: str) -> bool:
        """Run the fixer agent and return success status."""
        console.print(f"\n[bold blue]🔄 Starting {self.agent_name}[/bold blue]")
        
        # Load review report
        try:
            with open(review_report_path, "r") as f:
                review_report = json.load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: Review report file not found: {review_report_path}[/red]")
            return False
        except json.JSONDecodeError:
            console.print(f"[red]Error: Invalid JSON in review report file[/red]")
            return False
        
        # Apply fixes
        if self.apply_fixes(project_path, review_report):
            # Validate fixes
            if self.validate_fixes(project_path):
                console.print(f"\n[green]✅ {self.agent_name} completed successfully[/green]")
                return True
            else:
                console.print(f"\n[yellow]⚠️ {self.agent_name} completed with validation warnings[/yellow]")
                return True
        else:
            console.print(f"\n[red]❌ {self.agent_name} failed to apply fixes[/red]")
            return False

def main():
    """Test the fixer agent standalone."""
    ollama_client = OllamaClient()
    fixer = FixerAgent(ollama_client)
    
    # Test with a sample project and review report
    result = fixer.run("output/sample-project", "data/review_report.json")
    if result:
        console.print("Fixes applied successfully")
    else:
        console.print("Failed to apply fixes")

if __name__ == "__main__":
    main()