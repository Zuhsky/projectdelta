import json
import os
from typing import Dict, Any, List
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

console = Console()

class CLIWizard:
    def __init__(self):
        self.project_data = {}
    
    def display_welcome(self):
        """Display welcome message and introduction."""
        welcome_text = """
🚀 Welcome to the AI Development Team Orchestrator!

This tool will simulate a complete software development team to build your web application.
We'll ask you a series of questions to understand your project requirements,
then our AI agents will work together to create a production-ready website.

Our Team:
• 📋 Planner (Product Manager) - Gathers requirements
• 🏗️  Builder (Full-Stack Developer) - Creates the application  
• 🔍 Reviewer (Lead Engineer) - Reviews code quality
• 🔧 Fixer (Senior Debugger) - Fixes issues and optimizes
• ✅ Finalizer (QA Engineer) - Tests and documents
• 🚀 Git Pusher (DevOps) - Deploys and manages Git

Let's start building your dream project!
        """
        console.print(Panel(welcome_text, title="🎯 AI Development Team", border_style="blue"))
    
    def ask_basic_info(self):
        """Collect basic project information."""
        console.print("\n[bold blue]📝 Basic Project Information[/bold blue]")
        
        self.project_data["project_name"] = Prompt.ask(
            "What would you like to name your project?",
            default="my-awesome-app"
        ).lower().replace(" ", "-")
        
        self.project_data["description"] = Prompt.ask(
            "Describe your project in one sentence",
            default="A web application"
        )
        
        self.project_data["purpose"] = Prompt.ask(
            "What is the main purpose of this website?",
            choices=["saas", "ecommerce", "portfolio", "blog", "marketplace", "social", "other"],
            default="saas"
        )
        
        if self.project_data["purpose"] == "other":
            self.project_data["custom_purpose"] = Prompt.ask("Please describe the purpose")
    
    def ask_target_audience(self):
        """Collect target audience information."""
        console.print("\n[bold blue]👥 Target Audience[/bold blue]")
        
        self.project_data["target_audience"] = Prompt.ask(
            "Who is your primary target audience?",
            choices=["businesses", "consumers", "developers", "students", "professionals", "other"],
            default="consumers"
        )
        
        if self.project_data["target_audience"] == "other":
            self.project_data["custom_audience"] = Prompt.ask("Please describe your target audience")
        
        self.project_data["user_personas"] = Prompt.ask(
            "Describe your typical user in a few words",
            default="Tech-savvy professionals aged 25-45"
        )
    
    def ask_design_preferences(self):
        """Collect design and UI preferences."""
        console.print("\n[bold blue]🎨 Design Preferences[/bold blue]")
        
        self.project_data["design_style"] = Prompt.ask(
            "What design style do you prefer?",
            choices=["modern", "minimal", "corporate", "creative", "dark", "colorful"],
            default="modern"
        )
        
        self.project_data["color_scheme"] = Prompt.ask(
            "Preferred color scheme",
            choices=["blue", "green", "purple", "red", "neutral", "custom"],
            default="blue"
        )
        
        if self.project_data["color_scheme"] == "custom":
            self.project_data["custom_colors"] = Prompt.ask("Describe your preferred colors")
        
        self.project_data["mobile_first"] = Confirm.ask(
            "Should this be mobile-first design?",
            default=True
        )
    
    def ask_features_and_pages(self):
        """Collect information about required features and pages."""
        console.print("\n[bold blue]📋 Features & Pages[/bold blue]")
        
        # Core pages
        pages_table = Table(title="Select the pages you need")
        pages_table.add_column("Page", style="cyan")
        pages_table.add_column("Include?", style="green")
        
        common_pages = [
            "Landing/Home", "About", "Contact", "Pricing", "Blog", 
            "Dashboard", "User Profile", "Settings", "FAQ", "Terms of Service"
        ]
        
        selected_pages = []
        for page in common_pages:
            include = Confirm.ask(f"Include {page} page?", default=page in ["Landing/Home", "About", "Contact"])
            if include:
                selected_pages.append(page)
        
        self.project_data["pages"] = selected_pages
        
        # Authentication
        self.project_data["auth_required"] = Confirm.ask(
            "Do you need user authentication (login/register)?",
            default=True
        )
        
        if self.project_data["auth_required"]:
            self.project_data["auth_methods"] = Prompt.ask(
                "What authentication methods?",
                choices=["email", "social", "both"],
                default="email"
            )
        
        # Core features
        feature_categories = {
            "user_management": "User profiles and management",
            "payments": "Payment processing (Stripe/PayPal)",
            "search": "Search functionality",
            "notifications": "Email/push notifications",
            "analytics": "User analytics dashboard",
            "api": "RESTful API endpoints",
            "admin": "Admin panel",
            "chat": "Real-time chat/messaging",
            "file_upload": "File upload functionality",
            "subscription": "Subscription management"
        }
        
        selected_features = []
        console.print("\n[bold]Core Features:[/bold]")
        for key, description in feature_categories.items():
            if Confirm.ask(f"Include {description}?", default=False):
                selected_features.append(key)
        
        self.project_data["features"] = selected_features
    
    def ask_technical_requirements(self):
        """Collect technical requirements and preferences."""
        console.print("\n[bold blue]⚙️ Technical Requirements[/bold blue]")
        
        self.project_data["database_type"] = Prompt.ask(
            "What database do you prefer?",
            choices=["postgresql", "mysql", "sqlite", "mongodb"],
            default="postgresql"
        )
        
        self.project_data["deployment_preference"] = Prompt.ask(
            "Where would you like to deploy?",
            choices=["vercel", "netlify", "aws", "digitalocean", "other"],
            default="vercel"
        )
        
        self.project_data["third_party_integrations"] = []
        integrations = {
            "stripe": "Stripe (payments)",
            "sendgrid": "SendGrid (email)",
            "google_analytics": "Google Analytics",
            "cloudinary": "Cloudinary (image hosting)",
            "auth0": "Auth0 (authentication)",
            "redis": "Redis (caching)"
        }
        
        console.print("\n[bold]Third-party Integrations:[/bold]")
        for key, description in integrations.items():
            if Confirm.ask(f"Include {description}?", default=False):
                self.project_data["third_party_integrations"].append(key)
    
    def ask_scale_and_performance(self):
        """Collect scalability and performance requirements."""
        console.print("\n[bold blue]📈 Scale & Performance[/bold blue]")
        
        self.project_data["expected_users"] = Prompt.ask(
            "Expected number of users in first year?",
            choices=["<100", "100-1000", "1000-10000", "10000+"],
            default="100-1000"
        )
        
        self.project_data["performance_priority"] = Prompt.ask(
            "What's most important?",
            choices=["speed", "seo", "user_experience", "scalability"],
            default="user_experience"
        )
        
        self.project_data["seo_important"] = Confirm.ask(
            "Is SEO optimization important?",
            default=True
        )
        
        self.project_data["analytics_required"] = Confirm.ask(
            "Do you need built-in analytics?",
            default=False
        )
    
    def ask_content_management(self):
        """Ask about content management needs."""
        console.print("\n[bold blue]📝 Content Management[/bold blue]")
        
        self.project_data["cms_needed"] = Confirm.ask(
            "Do you need a content management system (CMS)?",
            default=False
        )
        
        if self.project_data["cms_needed"]:
            self.project_data["cms_type"] = Prompt.ask(
                "What type of content will you manage?",
                choices=["blog_posts", "products", "pages", "media", "all"],
                default="blog_posts"
            )
        
        self.project_data["multi_language"] = Confirm.ask(
            "Do you need multi-language support?",
            default=False
        )
    
    def ask_additional_requirements(self):
        """Collect any additional requirements."""
        console.print("\n[bold blue]➕ Additional Requirements[/bold blue]")
        
        self.project_data["custom_requirements"] = Prompt.ask(
            "Any specific features or requirements not covered?",
            default=""
        )
        
        self.project_data["budget_tier"] = Prompt.ask(
            "What's your budget tier for third-party services?",
            choices=["free", "low", "medium", "high"],
            default="free"
        )
        
        self.project_data["timeline"] = Prompt.ask(
            "When do you need this completed?",
            choices=["asap", "1-2_weeks", "1_month", "flexible"],
            default="flexible"
        )
    
    def display_summary(self):
        """Display a summary of collected requirements."""
        console.print("\n[bold green]📋 Project Summary[/bold green]")
        
        summary_table = Table(title=f"Project: {self.project_data['project_name']}")
        summary_table.add_column("Aspect", style="cyan")
        summary_table.add_column("Details", style="white")
        
        summary_table.add_row("Description", self.project_data["description"])
        summary_table.add_row("Purpose", self.project_data["purpose"])
        summary_table.add_row("Target Audience", self.project_data["target_audience"])
        summary_table.add_row("Design Style", self.project_data["design_style"])
        summary_table.add_row("Pages", ", ".join(self.project_data["pages"]))
        summary_table.add_row("Features", ", ".join(self.project_data["features"]))
        summary_table.add_row("Database", self.project_data["database_type"])
        summary_table.add_row("Expected Users", self.project_data["expected_users"])
        
        console.print(summary_table)
    
    def save_project_spec(self, output_path: str = "data/project_spec.json"):
        """Save the collected data to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add some derived technical specifications
        self.project_data["tech_stack"] = {
            "frontend": "Next.js 14 with TypeScript",
            "styling": "Tailwind CSS",
            "backend": "Next.js API routes",
            "database": self.project_data["database_type"],
            "orm": "Prisma",
            "deployment": self.project_data["deployment_preference"]
        }
        
        # Add architecture recommendations
        self.project_data["architecture"] = {
            "pattern": "MVC with API-first design",
            "components": "Atomic design system",
            "state_management": "React Context + useState",
            "api_design": "RESTful with OpenAPI docs",
            "security": "JWT tokens, CORS, rate limiting",
            "testing": "Jest + React Testing Library"
        }
        
        with open(output_path, 'w') as f:
            json.dump(self.project_data, f, indent=2)
        
        console.print(f"\n[green]✅ Project specification saved to {output_path}[/green]")
        return output_path
    
    def run_wizard(self):
        """Run the complete wizard flow."""
        try:
            self.display_welcome()
            
            self.ask_basic_info()
            self.ask_target_audience()
            self.ask_design_preferences()
            self.ask_features_and_pages()
            self.ask_technical_requirements()
            self.ask_scale_and_performance()
            self.ask_content_management()
            self.ask_additional_requirements()
            
            self.display_summary()
            
            if Confirm.ask("\nProceed with this configuration?", default=True):
                return self.save_project_spec()
            else:
                console.print("[yellow]Configuration cancelled. You can run the wizard again.[/yellow]")
                return None
                
        except KeyboardInterrupt:
            console.print("\n[red]Wizard cancelled by user.[/red]")
            return None
        except Exception as e:
            console.print(f"\n[red]An error occurred: {e}[/red]")
            return None

def main():
    """Main function for testing the wizard standalone."""
    wizard = CLIWizard()
    wizard.run_wizard()

if __name__ == "__main__":
    main()