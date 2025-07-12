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
```

### 2. **Start Ollama (Important for RunPod containers)**
```bash
# Start Ollama in background (RunPod containers don't use systemd)
nohup ollama serve > ollama.log 2>&1 &

# Wait for Ollama to start
sleep 10

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### 3. **Pull Large Models (30B+)**
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

### 4. **Set Environment Variables**
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

### 5. **Install Python Dependencies**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

### 6. **Run the Orchestrator**
```bash
# Test GPU setup
python test_gpu_setup.py

# Run the main orchestrator
python main.py

# Or run with specific project
python main.py --project-name "my-nextjs-app" --description "A modern e-commerce platform"
```

### 7. **Monitor Performance**
```bash
# Check GPU usage
nvidia-smi

# Monitor Ollama logs
tail -f ollama.log

# Check orchestrator logs
tail -f orchestrator.log

# Monitor system resources
htop
```

### 8. **Useful Commands for RunPod**
```bash
# Start Ollama (if not running)
nohup ollama serve > ollama.log 2>&1 &

# Stop Ollama
pkill ollama

# Check if Ollama is running
curl http://localhost:11434/api/tags

# View Ollama logs
tail -f ollama.log

# Restart Ollama
pkill ollama && sleep 2 && nohup ollama serve > ollama.log 2>&1 &
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
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
nohup ollama serve > ollama.log 2>&1 &

# Check Ollama logs
tail -f ollama.log

# Verify GPU access
nvidia-smi

# Kill and restart Ollama if needed
pkill ollama
sleep 2
nohup ollama serve > ollama.log 2>&1 &
```

## 🎯 **One-Command Setup Script**

You can also use the provided setup script:
```bash
chmod +x scripts/setup_runpod.sh
./scripts/setup_runpod.sh
```

This script will automatically:
- Install all dependencies
- Setup Ollama (without systemd)
- Configure environment
- Pull a default model
- Test the setup

## 🚨 **Important Notes for RunPod Containers**

- **No systemd**: RunPod containers don't use systemd, so we start Ollama manually
- **Background process**: Use `nohup` to keep Ollama running in background
- **Port 11434**: Make sure this port is accessible in your RunPod configuration
- **GPU access**: Verify GPU is available with `nvidia-smi`
- **Memory management**: Monitor memory usage with `htop` and `nvidia-smi`
