# 🤖 AI Development Team Orchestrator - GPU Optimized

A comprehensive Python CLI tool that simulates a full-stack software development team of 6 AI agents, each representing a real professional role in a web development company. This system generates complex, scalable, production-grade websites using local Ollama models.

**🚀 Now optimized for RunPod GPU environment with 30B+ models!**

## 🎯 Overview

Transform a simple one-line idea into a complete, production-ready web application - just like hiring a top-tier development agency! The AI Development Team Orchestrator coordinates 6 specialized AI agents to deliver enterprise-grade web applications with comprehensive documentation, testing, and deployment configurations.

### 🏗️ What You Get

- **Complete Next.js 14 Application** with TypeScript, Tailwind CSS, and Prisma
- **Production-Ready Codebase** following industry best practices
- **Comprehensive Documentation** including API docs, deployment guides, and user manuals
- **Git Repository** with organized commit history and CI/CD configurations
- **Deployment Configurations** for Vercel, Netlify, Docker, and more
- **Quality Assurance** with code review, security audit, and performance optimization

## 👥 Meet Your AI Development Team

| Role | Agent | Model | GPU Optimization |
|------|-------|-------|-----------------|
| 📋 **Product Manager** | Planner | Mixtral 8x7B | High-context planning |
| 🏗️ **Full-Stack Developer** | Builder | CodeLlama 70B | Advanced code generation |
| 🔍 **Lead Engineer** | Reviewer | DeepSeek-Coder:33b | Code quality analysis |
| 🔧 **Senior Debugger** | Fixer | DeepSeek-Coder:33b | Bug fixing & optimization |
| ✅ **QA Engineer** | Finalizer | Llama2 70B | Documentation & testing |
| 🚀 **DevOps Engineer** | Git Pusher | Llama2 70B | Deployment & Git |

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed on your system
- **Ollama** installed and running locally
- **Node.js 18+** (for running generated applications)
- **Git** (for version control)
- **GPU with 12GB+ VRAM** (recommended: RTX 3090/4090 or A40/A100)
- **RunPod environment** (for optimal performance)

### Installation

1. **Clone or download the orchestrator:**
   ```bash
   # Download the orchestrator code to your machine
   cd orchestrator/
   ```

2. **Run the GPU-optimized setup script:**
   ```bash
   chmod +x scripts/setup_runpod.sh
   ./scripts/setup_runpod.sh
   ```
   
   This will:
   - Install Python dependencies with GPU support
   - Download 30B+ Ollama models (`llama2:70b-chat`, `deepseek-coder:33b`, `codellama:70b-instruct`, `mixtral:8x7b-instruct`)
   - Configure GPU optimization
   - Set up RunPod environment
   - Verify all prerequisites

3. **Start building your app:**
   ```bash
   python main.py
   ```

### Alternative Manual Setup

If the setup script doesn't work on your system:

```bash
# Install Python dependencies with GPU support
pip install -r requirements.txt

# Ensure Ollama is running with GPU optimization
ollama serve

# Pull 30B+ models for GPU acceleration
ollama pull llama2:70b-chat
ollama pull deepseek-coder:33b
ollama pull codellama:70b-instruct
ollama pull mixtral:8x7b-instruct

# Run the GPU-optimized orchestrator
python main.py
```

## 🎮 Usage

### Interactive Mode (Recommended)

```bash
python main.py
```

The interactive wizard will guide you through:
1. **Project Discovery** - 12+ detailed questions about your vision
2. **Requirements Analysis** - AI extracts technical specifications
3. **Development Pipeline** - Watch your team build the application
4. **Quality Assurance** - Automated testing and optimization
5. **Deployment Setup** - Git repository and deployment configs

### Command Line Options

```bash
# Show help
python main.py --help

# Skip wizard, use existing spec
python main.py --skip-wizard

# Use custom specification file
python main.py --spec-file my-project.json

# Show version
python main.py --version
```

## 📋 Example Project Ideas

The orchestrator can build various types of applications:

- **SaaS Platforms**: "Build me a project management tool for teams"
- **E-commerce**: "Create an online marketplace for handmade items"
- **Social Apps**: "Build a social network for developers"
- **Business Tools**: "Create a CRM system for small businesses"
- **Content Platforms**: "Build a blog platform with user authentication"
- **Portfolio Sites**: "Create a portfolio website for designers"

## 🏗️ Architecture

### Agent Workflow

```
User Input → Planner → Builder → Reviewer → Fixer → Finalizer → Git Pusher
     ↓           ↓         ↓         ↓        ↓          ↓          ↓
Requirements → Tech Spec → Code → Review → Fixes → Docs → Deployment
```

### Tech Stack

**Generated Applications Use:**
- **Frontend**: Next.js 14 with App Router, TypeScript, Tailwind CSS
- **Backend**: Next.js API Routes, Prisma ORM
- **Database**: PostgreSQL (configurable)
- **Authentication**: NextAuth.js (configurable)
- **Deployment**: Vercel/Netlify ready with CI/CD

**Orchestrator Built With:**
- **Python 3.8+** with Rich CLI interface
- **Ollama** for local AI model inference
- **DeepSeek Models** for code generation and analysis

## 📁 Project Structure

```
orchestrator/
├── main.py                 # Main orchestration runner
├── cli_wizard.py           # Interactive user input collector
├── setup.sh               # Environment setup script
├── requirements.txt       # Python dependencies
├── agents/                 # AI agent implementations
│   ├── planner.py         # Product Manager
│   ├── builder.py         # Full-Stack Developer
│   ├── reviewer.py        # Lead Engineer
│   ├── fixer.py           # Senior Debugger
│   ├── finalizer.py       # QA Engineer
│   └── git_pusher.py      # DevOps Engineer
├── prompts/               # Agent system prompts
│   ├── planner_prompt.txt
│   ├── builder_prompt.txt
│   ├── reviewer_prompt.txt
│   ├── fixer_prompt.txt
│   ├── finalizer_prompt.txt
│   └── git_pusher_prompt.txt
├── utils/
│   └── ollama_client.py   # Ollama API interface
├── data/                  # Generated specifications
├── output/                # Generated projects
└── README.md
```

## 🔧 Configuration

### Ollama Models

The orchestrator uses two models:
- **llama2:7b-chat**: For planning, analysis, and documentation
- **deepseek-coder:33b**: For code generation, review, and debugging

### Customizing Agents

Each agent's behavior can be customized by editing the prompt files in `prompts/`:

```bash
# Edit the planner's behavior
nano prompts/planner_prompt.txt

# Modify the builder's code generation style
nano prompts/builder_prompt.txt
```

### Environment Variables

Create a `.env` file for custom configurations:

```env
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_OUTPUT_DIR=./output
LOG_LEVEL=INFO
```

## 📊 Output Quality

### Code Quality Standards

- ✅ **TypeScript Strict Mode** - Full type safety
- ✅ **ESLint Configuration** - Code consistency
- ✅ **Responsive Design** - Mobile-first approach
- ✅ **Accessibility** - WCAG AA compliance
- ✅ **SEO Optimization** - Meta tags, sitemap, structured data
- ✅ **Security** - Input validation, CORS, rate limiting
- ✅ **Performance** - Code splitting, lazy loading, optimization

### Documentation Included

- 📚 **README.md** - Setup and overview
- 🔗 **API Documentation** - Complete endpoint reference
- 🚀 **Deployment Guide** - Multi-platform deployment
- 👥 **User Guide** - End-user documentation
- 🔧 **Environment Setup** - Configuration templates

## 🚀 Deployment

Generated projects are ready for immediate deployment:

### Vercel (Recommended)
```bash
cd output/your-project
npm install
vercel --prod
```

### Netlify
```bash
cd output/your-project
npm install
npm run build
# Deploy via Netlify CLI or dashboard
```

### Docker
```bash
cd output/your-project
docker build -t my-app .
docker run -p 3000:3000 my-app
```

## 🛠️ Troubleshooting

### Common Issues

**Ollama Connection Failed**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve
```

**Model Not Found**
```bash
# Pull required models manually
ollama pull llama2:7b-chat
ollama pull deepseek-coder:33b
```

**Python Dependencies**
```bash
# Install in virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

**Generation Takes Too Long**
- Ensure you have sufficient RAM (8GB+ recommended)
- Close other applications using GPU/CPU
- Use smaller models if available

### Debug Mode

Enable verbose logging:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

## 📈 Performance Tips

### System Requirements

**Minimum:**
- RAM: 8GB
- CPU: 4 cores
- Storage: 10GB free space

**Recommended:**
- RAM: 16GB+
- CPU: 8+ cores or GPU acceleration
- Storage: 20GB+ SSD

### Optimization

- Close unnecessary applications during generation
- Use GPU acceleration if available with Ollama
- Ensure stable internet connection for model downloads

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. **Report Issues** - Found a bug? Open an issue
2. **Improve Prompts** - Better prompts = better output
3. **Add Features** - New agent types, deployment targets
4. **Documentation** - Help others understand and use the tool

### Development Setup

```bash
# Clone for development
git clone <repository-url>
cd orchestrator

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama Team** - For making local AI models accessible
- **DeepSeek** - For providing excellent coding models
- **Next.js Team** - For the amazing React framework
- **Open Source Community** - For the tools and libraries that make this possible

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ai-dev-team/orchestrator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ai-dev-team/orchestrator/discussions)
- **Documentation**: [Wiki](https://github.com/ai-dev-team/orchestrator/wiki)

---

**Built with ❤️ by the AI Development Team Orchestrator**

*Transform your ideas into production-ready applications with the power of AI*# Fixed model availability issues - tested with ollama client
