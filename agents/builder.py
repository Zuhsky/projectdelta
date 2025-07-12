import json
import os
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TaskID

from utils.ollama_client import OllamaClient

console = Console()

class BuilderAgent:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.model = "deepseek-coder:33b"
        self.agent_name = "Builder (Full-Stack Developer)"
        
    def load_prompt(self) -> str:
        """Load the builder prompt from file."""
        try:
            with open("prompts/builder_prompt.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            console.print("[red]Error: builder_prompt.txt not found[/red]")
            return ""
    
    def create_project_structure(self, project_name: str, output_dir: str = "output") -> str:
        """Create the basic project structure."""
        project_path = os.path.join(output_dir, project_name)
        
        # Remove existing directory if it exists
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        
        # Create project directory
        os.makedirs(project_path, exist_ok=True)
        
        console.print(f"[green]✅ Created project directory: {project_path}[/green]")
        return project_path
    
    def generate_codebase(self, technical_spec: Dict[str, Any], project_path: str) -> bool:
        """Generate the complete codebase based on technical specifications."""
        console.print(Panel(
            f"🏗️ {self.agent_name} is building the application...",
            title="Development Phase",
            border_style="green"
        ))
        
        # Load system prompt
        system_prompt = self.load_prompt()
        if not system_prompt:
            return False
        
        # Create user prompt with technical specification
        user_prompt = f"""
Please implement a complete Next.js 14 web application based on the following technical specification:

TECHNICAL SPECIFICATION:
{json.dumps(technical_spec, indent=2)}

Requirements:
1. Create a production-ready Next.js 14 application with TypeScript
2. Use Tailwind CSS for styling
3. Implement Prisma for database management
4. Follow the app router pattern
5. Include all specified features and pages
6. Implement proper authentication if required
7. Create responsive, accessible components
8. Include proper error handling and loading states
9. Follow best practices for security and performance
10. Structure code following atomic design principles

Please provide the complete file structure and implementation. Include all necessary configuration files, components, pages, API routes, database schemas, and utilities.

Generate a production-ready codebase that can be deployed immediately.
        """
        
        console.print("[yellow]Generating complete application codebase...[/yellow]")
        console.print("[dim]This may take several minutes for complex applications...[/dim]")
        
        # Generate codebase using Ollama
        response = self.ollama_client.generate(
            model=self.model,
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.2,  # Low temperature for consistent code generation
            max_tokens=16000  # Large context for complete codebase
        )
        
        if not response:
            console.print("[red]Failed to generate codebase[/red]")
            return False
        
        # Parse and create files from response
        return self.parse_and_create_files(response, project_path)
    
    def parse_and_create_files(self, response: str, project_path: str) -> bool:
        """Parse the AI response and create the actual files."""
        try:
            console.print("[yellow]Parsing response and creating files...[/yellow]")
            
            # Split response by file markers
            files_created = 0
            current_file = None
            current_content = []
            
            lines = response.split('\n')
            
            with Progress() as progress:
                task = progress.add_task("Creating files...", total=100)
                
                for line in lines:
                    if line.startswith('FILE: '):
                        # Save previous file if exists
                        if current_file and current_content:
                            self.create_file(project_path, current_file, '\n'.join(current_content))
                            files_created += 1
                            progress.update(task, advance=2)
                        
                        # Start new file
                        current_file = line.replace('FILE: ', '').strip()
                        current_content = []
                        
                    elif line.startswith('PURPOSE: '):
                        # Skip purpose line
                        continue
                        
                    elif line.strip() == '---':
                        # Skip separator lines
                        continue
                        
                    elif current_file:
                        # Add content to current file
                        current_content.append(line)
                
                # Save last file
                if current_file and current_content:
                    self.create_file(project_path, current_file, '\n'.join(current_content))
                    files_created += 1
                
                progress.update(task, completed=100)
            
            if files_created == 0:
                # Fallback: try to extract files using different patterns
                files_created = self.extract_files_fallback(response, project_path)
            
            console.print(f"[green]✅ Created {files_created} files[/green]")
            return files_created > 0
            
        except Exception as e:
            console.print(f"[red]Error parsing response: {e}[/red]")
            return False
    
    def extract_files_fallback(self, response: str, project_path: str) -> int:
        """Fallback method to extract files from response."""
        files_created = 0
        
        # Try to find code blocks and file references
        import re
        
        # Pattern to match file paths and code blocks
        file_pattern = r'(?:FILE:|```|\*\*|##)\s*([a-zA-Z0-9\-_/.]+\.[a-zA-Z0-9]+)\s*(?:```|:|$)'
        code_block_pattern = r'```(?:typescript|javascript|json|css|tsx|jsx)?\n(.*?)```'
        
        file_matches = re.findall(file_pattern, response, re.IGNORECASE)
        code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
        
        # Try to match files with code blocks
        for i, file_path in enumerate(file_matches[:len(code_blocks)]):
            if i < len(code_blocks):
                content = code_blocks[i].strip()
                if content:
                    self.create_file(project_path, file_path, content)
                    files_created += 1
        
        # If no matches, create basic project structure
        if files_created == 0:
            files_created = self.create_basic_structure(project_path)
        
        return files_created
    
    def create_basic_structure(self, project_path: str) -> int:
        """Create a basic Next.js project structure as fallback."""
        console.print("[yellow]Creating basic Next.js structure as fallback...[/yellow]")
        
        basic_files = {
            "package.json": """{
  "name": "generated-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.0.0",
    "react": "^18",
    "react-dom": "^18",
    "typescript": "^5",
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.0.1",
    "postcss": "^8"
  }
}""",
            "next.config.js": """/** @type {import('next').NextConfig} */
const nextConfig = {}

module.exports = nextConfig""",
            "tsconfig.json": """{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}""",
            "tailwind.config.js": """/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}""",
            "app/layout.tsx": """import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Generated App',
  description: 'Generated by AI Development Team',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}""",
            "app/page.tsx": """export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold">Welcome to Your Generated App</h1>
        <p className="mt-4 text-lg">
          This application was generated by the AI Development Team Orchestrator.
        </p>
      </div>
    </main>
  )
}""",
            "app/globals.css": """@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --foreground-rgb: 0, 0, 0;
  --background-start-rgb: 214, 219, 220;
  --background-end-rgb: 255, 255, 255;
}

@media (prefers-color-scheme: dark) {
  :root {
    --foreground-rgb: 255, 255, 255;
    --background-start-rgb: 0, 0, 0;
    --background-end-rgb: 0, 0, 0;
  }
}

body {
  color: rgb(var(--foreground-rgb));
  background: linear-gradient(
      to bottom,
      transparent,
      rgb(var(--background-end-rgb))
    )
    rgb(var(--background-start-rgb));
}""",
            "README.md": """# Generated Application

This application was generated by the AI Development Team Orchestrator.

## Getting Started

First, install dependencies:

```bash
npm install
```

Then, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Tech Stack

- Next.js 14
- TypeScript
- Tailwind CSS
- React

## Features

This application includes the features specified in your project requirements.
"""
        }
        
        files_created = 0
        for file_path, content in basic_files.items():
            self.create_file(project_path, file_path, content)
            files_created += 1
        
        return files_created
    
    def create_file(self, project_path: str, file_path: str, content: str):
        """Create a file with the given content."""
        try:
            full_path = os.path.join(project_path, file_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write file content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            console.print(f"[dim]Created: {file_path}[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error creating file {file_path}: {e}[/red]")
    
    def validate_project_structure(self, project_path: str) -> bool:
        """Validate that the project has the basic required files."""
        required_files = [
            "package.json",
            "tsconfig.json",
            "next.config.js"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(project_path, file)):
                missing_files.append(file)
        
        if missing_files:
            console.print(f"[yellow]Warning: Missing required files: {', '.join(missing_files)}[/yellow]")
            return False
        
        console.print("[green]✅ Project structure validation passed[/green]")
        return True
    
    def run(self, technical_spec_path: str) -> Optional[str]:
        """Run the builder agent and return the path to the generated project."""
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
        
        # Get project name
        project_name = technical_spec.get("project_overview", {}).get("name", "generated-app")
        project_name = project_name.lower().replace(" ", "-").replace("_", "-")
        
        # Create project structure
        project_path = self.create_project_structure(project_name)
        
        # Generate codebase
        if self.generate_codebase(technical_spec, project_path):
            # Validate project structure
            if self.validate_project_structure(project_path):
                console.print(f"\n[green]✅ {self.agent_name} completed successfully[/green]")
                return project_path
            else:
                console.print(f"\n[yellow]⚠️ {self.agent_name} completed with warnings[/yellow]")
                return project_path
        else:
            console.print(f"\n[red]❌ {self.agent_name} failed to generate codebase[/red]")
            return None

def main():
    """Test the builder agent standalone."""
    ollama_client = OllamaClient()
    builder = BuilderAgent(ollama_client)
    
    # Test with a sample technical spec
    result = builder.run("data/technical_spec.json")
    if result:
        console.print(f"Project generated at: {result}")
    else:
        console.print("Failed to generate project")

if __name__ == "__main__":
    main()