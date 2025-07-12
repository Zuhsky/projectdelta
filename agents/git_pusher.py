import json
import os
import subprocess
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.panel import Panel

from utils.ollama_client import OllamaClient

console = Console()

class GitPusherAgent:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.model = "deepseek-coder:33b"
        self.agent_name = "Git Pusher (DevOps)"
        
    def load_prompt(self) -> str:
        """Load the git pusher prompt from file."""
        try:
            with open("prompts/git_pusher_prompt.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            console.print("[red]Error: git_pusher_prompt.txt not found[/red]")
            return ""
    
    def initialize_git_repository(self, project_path: str) -> bool:
        """Initialize Git repository and create initial commit."""
        console.print(Panel(
            f"🚀 {self.agent_name} is setting up Git repository and deployment configurations...",
            title="Git & Deployment Setup",
            border_style="magenta"
        ))
        
        try:
            # Initialize git repository
            subprocess.run(["git", "init"], cwd=project_path, check=True, capture_output=True)
            console.print("[green]✅ Git repository initialized[/green]")
            
            # Create .gitignore if it doesn't exist
            gitignore_path = os.path.join(project_path, ".gitignore")
            if not os.path.exists(gitignore_path):
                self.create_gitignore(project_path)
            
            # Configure git settings
            subprocess.run(["git", "config", "user.name", "AI Development Team"], cwd=project_path, check=True)
            subprocess.run(["git", "config", "user.email", "ai-team@example.com"], cwd=project_path, check=True)
            
            # Add all files
            subprocess.run(["git", "add", "."], cwd=project_path, check=True)
            
            # Create initial commit
            subprocess.run(["git", "commit", "-m", "feat: initial project setup and implementation\n\n- Complete Next.js 14 application structure\n- TypeScript configuration\n- Tailwind CSS setup\n- Component architecture\n- API routes implementation\n- Documentation and deployment configs"], cwd=project_path, check=True)
            
            console.print("[green]✅ Initial commit created[/green]")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Git initialization failed: {e}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Error during git setup: {e}[/red]")
            return False
    
    def create_gitignore(self, project_path: str):
        """Create a comprehensive .gitignore file."""
        gitignore_content = """# Dependencies
node_modules/
/.pnp
.pnp.js

# Testing
/coverage

# Next.js
/.next/
/out/

# Production
/build

# Misc
.DS_Store
*.pem

# Debug
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Local env files
.env*.local
.env

# Vercel
.vercel

# TypeScript
*.tsbuildinfo
next-env.d.ts

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs
*.log

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Optional npm cache directory
.npm

# Optional eslint cache
.eslintcache

# Prisma
prisma/migrations/

# OS generated files
Thumbs.db
"""
        
        gitignore_path = os.path.join(project_path, ".gitignore")
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        console.print("[dim]Created .gitignore[/dim]")
    
    def create_deployment_configs(self, project_path: str) -> bool:
        """Create deployment configuration files."""
        console.print("[yellow]Creating deployment configurations...[/yellow]")
        
        configs_created = 0
        
        # Create Vercel configuration
        vercel_config = {
            "framework": "nextjs",
            "buildCommand": "npm run build",
            "outputDirectory": ".next",
            "installCommand": "npm install",
            "env": {
                "NODE_ENV": "production"
            },
            "regions": ["iad1"],
            "functions": {
                "app/api/**/*.js": {
                    "maxDuration": 10
                }
            }
        }
        
        if self.create_file(project_path, "vercel.json", json.dumps(vercel_config, indent=2)):
            configs_created += 1
        
        # Create Netlify configuration
        netlify_config = """[build]
  publish = ".next"
  command = "npm run build"

[build.environment]
  NODE_ENV = "production"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    X-XSS-Protection = "1; mode=block"
"""
        
        if self.create_file(project_path, "netlify.toml", netlify_config):
            configs_created += 1
        
        # Create GitHub Actions CI/CD workflow
        github_workflow = """name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: [18.x, 20.x]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run type checking
      run: npm run type-check
    
    - name: Run linting
      run: npm run lint
    
    - name: Run tests
      run: npm run test --if-present
    
    - name: Build application
      run: npm run build

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Vercel
      uses: amondnet/vercel-action@v20
      with:
        vercel-token: ${{ secrets.VERCEL_TOKEN }}
        vercel-org-id: ${{ secrets.ORG_ID }}
        vercel-project-id: ${{ secrets.PROJECT_ID }}
        vercel-args: '--prod'
"""
        
        if self.create_file(project_path, ".github/workflows/ci.yml", github_workflow):
            configs_created += 1
        
        # Create Docker configuration
        dockerfile = """FROM node:18-alpine AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production

FROM node:18-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000

CMD ["node", "server.js"]
"""
        
        if self.create_file(project_path, "Dockerfile", dockerfile):
            configs_created += 1
        
        # Create docker-compose for development
        docker_compose = """version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - NEXTAUTH_SECRET=${NEXTAUTH_SECRET}
      - NEXTAUTH_URL=${NEXTAUTH_URL}
    depends_on:
      - db

  db:
    image: postgres:15
    restart: always
    environment:
      POSTGRES_DB: appdb
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
"""
        
        if self.create_file(project_path, "docker-compose.yml", docker_compose):
            configs_created += 1
        
        console.print(f"[green]✅ Created {configs_created} deployment configuration files[/green]")
        return configs_created > 0
    
    def create_file(self, project_path: str, file_path: str, content: str) -> bool:
        """Create a file with the given content."""
        try:
            full_path = os.path.join(project_path, file_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write file content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            console.print(f"[dim]Created: {file_path}[/dim]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error creating file {file_path}: {e}[/red]")
            return False
    
    def create_organized_commits(self, project_path: str) -> bool:
        """Create organized commit history by component."""
        console.print("[yellow]Creating organized commit history...[/yellow]")
        
        try:
            # Reset to initial commit for reorganization
            subprocess.run(["git", "reset", "--soft", "HEAD~1"], cwd=project_path, check=True, capture_output=True)
            subprocess.run(["git", "reset"], cwd=project_path, check=True, capture_output=True)
            
            # Commit configuration files
            config_files = ["package.json", "tsconfig.json", "next.config.js", "tailwind.config.js", ".gitignore"]
            for file in config_files:
                file_path = os.path.join(project_path, file)
                if os.path.exists(file_path):
                    subprocess.run(["git", "add", file], cwd=project_path, check=True)
            
            subprocess.run(["git", "commit", "-m", "feat: project configuration and setup\n\n- Next.js 14 configuration\n- TypeScript setup\n- Tailwind CSS configuration\n- Package dependencies"], cwd=project_path, check=True)
            
            # Commit components
            if os.path.exists(os.path.join(project_path, "components")):
                subprocess.run(["git", "add", "components/"], cwd=project_path, check=True)
                subprocess.run(["git", "commit", "-m", "feat: implement reusable UI components\n\n- Atomic design component structure\n- TypeScript interfaces\n- Responsive design\n- Accessibility features"], cwd=project_path, check=True)
            
            # Commit app directory (pages and layouts)
            if os.path.exists(os.path.join(project_path, "app")):
                subprocess.run(["git", "add", "app/"], cwd=project_path, check=True)
                subprocess.run(["git", "commit", "-m", "feat: implement application pages and layouts\n\n- Next.js 14 app router structure\n- Server and client components\n- SEO optimization\n- Loading and error states"], cwd=project_path, check=True)
            
            # Commit API routes
            api_path = os.path.join(project_path, "app", "api")
            if os.path.exists(api_path):
                subprocess.run(["git", "add", "app/api/"], cwd=project_path, check=True)
                subprocess.run(["git", "commit", "-m", "feat: implement API routes and backend logic\n\n- RESTful API endpoints\n- Input validation\n- Error handling\n- Authentication middleware"], cwd=project_path, check=True)
            
            # Commit database and utilities
            for dir_name in ["lib", "utils", "prisma"]:
                dir_path = os.path.join(project_path, dir_name)
                if os.path.exists(dir_path):
                    subprocess.run(["git", "add", f"{dir_name}/"], cwd=project_path, check=True)
            
            subprocess.run(["git", "commit", "-m", "feat: add database schema and utility functions\n\n- Prisma database setup\n- Utility functions\n- Type definitions\n- Helper modules"], cwd=project_path, check=True)
            
            # Commit documentation
            for doc_file in ["README.md", "docs/"]:
                doc_path = os.path.join(project_path, doc_file)
                if os.path.exists(doc_path):
                    subprocess.run(["git", "add", doc_file], cwd=project_path, check=True)
            
            subprocess.run(["git", "commit", "-m", "docs: add comprehensive project documentation\n\n- README with setup instructions\n- API documentation\n- Deployment guide\n- User guide"], cwd=project_path, check=True)
            
            # Commit deployment configurations
            deploy_files = ["vercel.json", "netlify.toml", ".github/", "Dockerfile", "docker-compose.yml", ".env.example"]
            for file in deploy_files:
                file_path = os.path.join(project_path, file)
                if os.path.exists(file_path):
                    subprocess.run(["git", "add", file], cwd=project_path, check=True)
            
            subprocess.run(["git", "commit", "-m", "ci: add deployment configurations and CI/CD\n\n- Vercel deployment config\n- GitHub Actions workflow\n- Docker containerization\n- Environment variables template"], cwd=project_path, check=True)
            
            # Add any remaining files
            result = subprocess.run(["git", "status", "--porcelain"], cwd=project_path, capture_output=True, text=True)
            if result.stdout.strip():
                subprocess.run(["git", "add", "."], cwd=project_path, check=True)
                subprocess.run(["git", "commit", "-m", "feat: add remaining project files\n\n- Additional assets and configurations\n- Final project structure"], cwd=project_path, check=True)
            
            console.print("[green]✅ Organized commit history created[/green]")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]Warning: Could not create organized commits: {e}[/yellow]")
            # Fall back to single commit
            try:
                subprocess.run(["git", "add", "."], cwd=project_path, check=True)
                subprocess.run(["git", "commit", "-m", "feat: complete project implementation"], cwd=project_path, check=True)
                return True
            except:
                return False
        except Exception as e:
            console.print(f"[red]Error creating commits: {e}[/red]")
            return False
    
    def setup_deployment_ready_state(self, project_path: str) -> Dict[str, Any]:
        """Ensure the project is deployment ready."""
        console.print("[yellow]Verifying deployment readiness...[/yellow]")
        
        deployment_status = {
            "git_initialized": False,
            "commits_created": False,
            "deployment_configs": False,
            "env_template": False,
            "documentation": False,
            "package_json_valid": False
        }
        
        # Check Git initialization
        if os.path.exists(os.path.join(project_path, ".git")):
            deployment_status["git_initialized"] = True
        
        # Check commits
        try:
            result = subprocess.run(["git", "log", "--oneline"], cwd=project_path, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                deployment_status["commits_created"] = True
        except:
            pass
        
        # Check deployment configs
        config_files = ["vercel.json", ".github/workflows/ci.yml"]
        if any(os.path.exists(os.path.join(project_path, f)) for f in config_files):
            deployment_status["deployment_configs"] = True
        
        # Check environment template
        if os.path.exists(os.path.join(project_path, ".env.example")):
            deployment_status["env_template"] = True
        
        # Check documentation
        if os.path.exists(os.path.join(project_path, "README.md")):
            deployment_status["documentation"] = True
        
        # Check package.json
        package_json_path = os.path.join(project_path, "package.json")
        if os.path.exists(package_json_path):
            try:
                with open(package_json_path, 'r') as f:
                    json.load(f)
                deployment_status["package_json_valid"] = True
            except:
                pass
        
        ready_count = sum(deployment_status.values())
        total_checks = len(deployment_status)
        
        console.print(f"[cyan]Deployment readiness: {ready_count}/{total_checks} checks passed[/cyan]")
        
        return deployment_status
    
    def run(self, project_path: str) -> bool:
        """Run the git pusher agent and return success status."""
        console.print(f"\n[bold blue]🔄 Starting {self.agent_name}[/bold blue]")
        
        success = True
        
        # Initialize Git repository
        if not self.initialize_git_repository(project_path):
            console.print("[yellow]⚠️ Git initialization had issues, continuing...[/yellow]")
            success = False
        
        # Create deployment configurations
        if not self.create_deployment_configs(project_path):
            console.print("[yellow]⚠️ Some deployment configs could not be created[/yellow]")
        
        # Create organized commits
        if not self.create_organized_commits(project_path):
            console.print("[yellow]⚠️ Could not create organized commit history[/yellow]")
        
        # Check deployment readiness
        deployment_status = self.setup_deployment_ready_state(project_path)
        
        if sum(deployment_status.values()) >= 4:  # At least 4 out of 6 checks should pass
            console.print(f"\n[bold green]🚀 Project is ready for deployment![/bold green]")
            console.print("[green]Next steps:[/green]")
            console.print("[green]1. Push to GitHub repository[/green]")
            console.print("[green]2. Connect to Vercel or your preferred platform[/green]")
            console.print("[green]3. Set up environment variables[/green]")
            console.print("[green]4. Deploy to production[/green]")
            
            console.print(f"\n[green]✅ {self.agent_name} completed successfully[/green]")
            return True
        else:
            console.print(f"\n[yellow]⚠️ {self.agent_name} completed with warnings[/yellow]")
            console.print("[yellow]Some deployment setup steps need manual attention[/yellow]")
            return success

def main():
    """Test the git pusher agent standalone."""
    ollama_client = OllamaClient()
    git_pusher = GitPusherAgent(ollama_client)
    
    # Test with a sample project
    result = git_pusher.run("output/sample-project")
    if result:
        console.print("Git setup completed successfully")
    else:
        console.print("Git setup completed with warnings")

if __name__ == "__main__":
    main()