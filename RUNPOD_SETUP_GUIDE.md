# 🚀 AI Development Team Orchestrator - RunPod Setup Guide

## 📋 RunPod Configuration & Setup

### Step 1: Choose the Right RunPod Template

**Recommended Configuration:**
- **Template**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **GPU**: RTX 4090 (24GB VRAM) - Best performance/cost ratio
- **Alternative**: RTX 3090 (24GB VRAM) - More budget-friendly
- **CPU**: 8+ vCPU cores
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ container disk

**Budget Options ($0.7/hour max):**
- RTX 3090: ~$0.34/hour
- RTX 4090: ~$0.79/hour
- A40: ~$0.60/hour

### Step 2: Create Your RunPod Instance

1. **Go to RunPod Console**: https://www.runpod.io/console/pods
2. **Click "Deploy"** → "GPU Pod"
3. **Select Template**: Choose PyTorch template above
4. **Configure**:
   ```
   Container Disk Size: 50GB
   Volume Disk Size: 20GB (optional for persistent storage)
   Expose HTTP Ports: 8888,3000,11434
   Environment Variables: None needed initially
   ```
5. **Deploy Pod**

## 🛠️ Installation & Setup Process

### Step 3: Access Your Pod

1. **Click "Connect"** on your pod
2. **Choose "Start Web Terminal"**
3. Wait for terminal to load

### Step 4: System Setup

```bash
# Update system
apt update && apt upgrade -y

# Install essential tools
apt install -y curl wget git build-essential python3-pip

# Install Node.js (for generated projects)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
apt install -y nodejs

# Verify installations
python3 --version
node --version
git --version
```

### Step 5: Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service in background
nohup ollama serve > ollama.log 2>&1 &

# Wait a moment for service to start
sleep 5

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Step 6: Download AI Models

```bash
# Pull required models (this will take 15-30 minutes)
echo "Downloading Llama2:7b-Chat..."
ollama pull llama2:7b-chat

echo "Downloading DeepSeek-Coder:33b..."
ollama pull deepseek-coder:33b

# Verify models are downloaded
ollama list
```

**Expected Output:**
```
NAME                    ID              SIZE      MODIFIED
llama2:7b-chat:latest  abc123...       3.8GB     2 minutes ago
deepseek-coder:33b      def456...       19GB      5 minutes ago
```

### Step 7: Clone and Setup the Orchestrator

```bash
# Clone the repository
git clone https://github.com/salsa1zg/ai-development-team-orchestrator.git

# Navigate to project
cd ai-development-team-orchestrator

# Install Python dependencies
pip3 install -r requirements.txt

# Make setup script executable
chmod +x setup.sh

# Run setup (this will verify everything)
./setup.sh
```

## 🎮 Using the AI Development Team Orchestrator

### Step 8: Start the Orchestrator

```bash
# Make sure you're in the project directory
cd ai-development-team-orchestrator

# Start the orchestrator
python3 main.py
```

### Step 9: Follow the Interactive Wizard

The tool will guide you through:

1. **Welcome Screen** - Shows your AI team
2. **Prerequisites Check** - Verifies Ollama and models
3. **Project Wizard** - 12+ questions about your app idea
4. **Development Pipeline** - Watch AI agents work:
   - 📋 Planner analyzes requirements
   - 🏗️ Builder creates the application
   - 🔍 Reviewer checks code quality
   - 🔧 Fixer applies optimizations
   - ✅ Finalizer adds documentation
   - 🚀 Git Pusher sets up deployment

### Step 10: Access Generated Projects

```bash
# View generated projects
ls -la output/

# Navigate to your project
cd output/your-project-name

# Install project dependencies
npm install

# Start development server
npm run dev
```

## 🌐 Port Configuration & Access

### RunPod Port Setup

When creating your pod, expose these ports:
- **11434**: Ollama API service
- **3000**: Next.js development server (for generated apps)
- **8888**: Alternative web access

### Access Your Generated App

1. **In RunPod Console**, click your pod
2. **Click "Connect"** → "Connect to HTTP Service [3000]"
3. **Your app will open** in a new browser tab

## 💡 RunPod-Specific Tips

### Optimize for Cost

```bash
# Monitor GPU usage
nvidia-smi

# Stop Ollama when not in use
pkill ollama

# Use smaller models for testing
ollama pull llama2:7b-chat  # Main chat model
```

### Persistent Storage Setup

If you want to keep projects between sessions:

```bash
# Create persistent directory
mkdir -p /workspace/persistent-projects

# Symlink to persistent storage
ln -s /workspace/persistent-projects ~/ai-development-team-orchestrator/output/persistent
```

### Background Processing

```bash
# Run orchestrator in background
nohup python3 main.py > orchestrator.log 2>&1 &

# Monitor progress
tail -f orchestrator.log
```

## 🔧 Troubleshooting Guide

### Common Issues & Solutions

**1. Ollama Not Starting**
```bash
# Check if port is in use
netstat -tulpn | grep 11434

# Kill existing process
pkill ollama

# Restart service
ollama serve &
```

**2. Out of Memory**
```bash
# Check memory usage
free -h

# Use smaller model
ollama pull llama2:7b-chat
```

**3. Model Download Fails**
```bash
# Check internet connection
ping google.com

# Download manually
ollama pull llama2:7b-chat --insecure
```

**4. Generated App Won't Start**
```bash
# Install Node.js if missing
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Check project directory
ls -la output/your-project/
cd output/your-project/
npm install
```

## 📊 Expected Performance

### Generation Times (RTX 4090):
- **Requirements Analysis**: 2-3 minutes
- **Code Generation**: 10-15 minutes
- **Code Review**: 3-5 minutes
- **Bug Fixes**: 5-8 minutes
- **Documentation**: 2-3 minutes
- **Total**: ~25-35 minutes

### Resource Usage:
- **GPU Memory**: 18-20GB during generation
- **System RAM**: 8-12GB
- **Disk Space**: 30-40GB (models + generated code)

## 🎯 Complete Workflow Example

```bash
# 1. SSH into RunPod (Use web terminal)

# 2. Quick setup
curl -fsSL https://ollama.ai/install.sh | sh && \
ollama serve & && \
sleep 5 && \
git clone https://github.com/salsa1zg/ai-development-team-orchestrator.git && \
cd ai-development-team-orchestrator && \
pip3 install -r requirements.txt

# 3. Download models
ollama pull llama2:7b-chat && ollama pull deepseek-coder:33b

# 4. Start orchestrator
python3 main.py

# 5. Follow wizard prompts
# (Answer questions about your app idea)

# 6. Wait for generation
# (Grab coffee, 25-35 minutes)

# 7. Test your app
cd output/your-project-name && npm install && npm run dev

# 8. Access via RunPod HTTP port 3000
```

## 🔒 Security & Best Practices

1. **Never store sensitive data** in RunPod instances
2. **Use volume storage** for important projects
3. **Stop instances** when not in use to save costs
4. **Monitor resource usage** to avoid overcharges
5. **Backup generated code** to GitHub or external storage

## 💰 Cost Management

**Estimated Costs:**
- **Setup (one-time)**: ~$2-3 (model downloads)
- **Per app generation**: ~$0.30-0.70 (30-40 minutes)
- **Development/testing**: ~$0.50/hour

**Cost-Saving Tips:**
- Use spot instances when available
- Stop pod immediately after generation
- Use smaller models for testing concepts
- Batch multiple app generations in one session

## 🚀 Quick Start Commands (Copy & Paste)

### One-Line Setup:
```bash
apt update && apt install -y curl git python3-pip nodejs npm && curl -fsSL https://ollama.ai/install.sh | sh && ollama serve & && sleep 5
```

### Clone and Install:
```bash
git clone https://github.com/salsa1zg/ai-development-team-orchestrator.git && cd ai-development-team-orchestrator && pip3 install -r requirements.txt
```

### Download Models:
```bash
ollama pull llama2:7b-chat && ollama pull deepseek-coder:33b
```

### Run Orchestrator:
```bash
python3 main.py
```

---

**Your AI Development Team Orchestrator is now ready to build production-grade applications on RunPod! 🚀**

For support: https://github.com/salsa1zg/ai-development-team-orchestrator/issues