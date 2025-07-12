# RunPod Setup Guide for AI Development Team Orchestrator

## 🚀 Quick Start Commands for RunPod

### 1. **Initial Setup**
```bash
# Update system and install dependencies
apt update && apt upgrade -y
apt install -y curl wget git python3 python3-pip python3-venv

# Clone your repository
git clone https://github.com/Zuhsky/projectdelta.git
cd projectdelta

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
systemctl start ollama
systemctl enable ollama
```

### 2. **Pull Large Models (30B+)**
```bash
# Pull your preferred 30B+ model (choose one)
ollama pull llama2:70b
# OR
ollama pull codellama:70b
# OR
ollama pull mixtral:8x7b
# OR
ollama pull qwen2.5:72b
```

### 3. **Set Environment Variables**
```bash
# Create environment file
cp env.example .env

# Edit with your settings
nano .env
```

**Recommended .env settings for RunPod:**
```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2:70b
OLLAMA_TIMEOUT=300
OLLAMA_MAX_TOKENS=4096
OLLAMA_TEMPERATURE=0.7

# GPU Optimization
USE_GPU=true
GPU_MEMORY_LIMIT=80
BATCH_SIZE=4
PARALLEL_REQUESTS=2

# Logging
LOG_LEVEL=INFO
LOG_FILE=orchestrator.log

# Performance
ENABLE_CACHING=true
CACHE_TTL=3600
ENABLE_MONITORING=true
```

### 4. **Install Python Dependencies**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

### 5. **Run the Orchestrator**
```bash
# Test GPU setup
python test_gpu_setup.py

# Run the main orchestrator
python main.py

# Or run with specific project
python main.py --project-name "my-nextjs-app" --description "A modern e-commerce platform"
```

### 6. **Monitor Performance**
```bash
# Check GPU usage
nvidia-smi

# Monitor Ollama logs
journalctl -u ollama -f

# Check orchestrator logs
tail -f orchestrator.log

# Monitor system resources
htop
```

### 7. **Optional: Setup Auto-restart**
```bash
# Create systemd service for auto-restart
nano /etc/systemd/system/ai-orchestrator.service
```

**Service file content:**
```ini
[Unit]
Description=AI Development Team Orchestrator
After=network.target ollama.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/projectdelta
Environment=PATH=/root/projectdelta/venv/bin
ExecStart=/root/projectdelta/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
systemctl daemon-reload
systemctl enable ai-orchestrator
systemctl start ai-orchestrator
```

## 📊 **Cost Optimization Tips for RunPod**

1. **Use Spot Instances** for development/testing
2. **Monitor GPU Usage** with `nvidia-smi`
3. **Stop instances** when not in use
4. **Use smaller models** for initial testing (7B-13B)
5. **Enable caching** to reduce API calls

## 🔧 **Troubleshooting**

If you encounter issues:
```bash
# Check Ollama status
systemctl status ollama

# Restart Ollama
systemctl restart ollama

# Check logs
journalctl -u ollama -n 50

# Verify GPU access
nvidia-smi
```

## 🎯 **One-Command Setup Script**

You can also use the provided setup script:
```bash
chmod +x scripts/setup_runpod.sh
./scripts/setup_runpod.sh
```

This script will automatically:
- Install all dependencies
- Setup Ollama
- Configure environment
- Pull a default model
- Test the setup