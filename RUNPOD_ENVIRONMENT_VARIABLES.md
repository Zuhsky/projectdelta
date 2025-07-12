# 🔧 RunPod Environment Variables for AI Development Team Orchestrator

## 📋 Recommended Environment Variables for Pod Template Override

### 🐍 **Python & Development Environment**
```
PYTHONPATH=/workspace
PYTHONUNBUFFERED=1
PIP_NO_CACHE_DIR=1
PIP_DISABLE_PIP_VERSION_CHECK=1
```

### 🤖 **Ollama Configuration**
```
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_ORIGINS=*
OLLAMA_DEBUG=1
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_NUM_PARALLEL=4
OLLAMA_MAX_QUEUE=512
```

### 🔧 **Git Configuration**
```
GIT_CONFIG_GLOBAL_USER_NAME=AI Development Team
GIT_CONFIG_GLOBAL_USER_EMAIL=ai-dev-team@runpod.local
GIT_CONFIG_GLOBAL_INIT_DEFAULTBRANCH=main
```

### 🚀 **Node.js & NPM Optimization**
```
NODE_ENV=development
NPM_CONFIG_CACHE=/tmp/.npm
NPM_CONFIG_PROGRESS=false
NPM_CONFIG_AUDIT=false
NPM_CONFIG_FUND=false
```

### ⚡ **Performance & GPU Optimization**
```
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
HF_HOME=/workspace/.cache/huggingface
TRANSFORMERS_CACHE=/workspace/.cache/transformers
```

### 🛠️ **Development Tools**
```
TERM=xterm-256color
COLORTERM=truecolor
DEBIAN_FRONTEND=noninteractive
TZ=UTC
```

### 📦 **AI Orchestrator Specific**
```
ORCHESTRATOR_OUTPUT_DIR=/workspace/output
ORCHESTRATOR_DATA_DIR=/workspace/data
ORCHESTRATOR_LOG_LEVEL=INFO
ORCHESTRATOR_AUTO_START=false
```

## 🎯 **Copy-Paste Ready Format for RunPod**

### **Environment Variables (Key=Value format):**
```
PYTHONPATH=/workspace
PYTHONUNBUFFERED=1
PIP_NO_CACHE_DIR=1
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_ORIGINS=*
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_NUM_PARALLEL=4
NODE_ENV=development
NPM_CONFIG_CACHE=/tmp/.npm
NPM_CONFIG_PROGRESS=false
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
GIT_CONFIG_GLOBAL_USER_NAME=AI Development Team
GIT_CONFIG_GLOBAL_USER_EMAIL=ai-dev-team@runpod.local
TERM=xterm-256color
DEBIAN_FRONTEND=noninteractive
ORCHESTRATOR_OUTPUT_DIR=/workspace/output
ORCHESTRATOR_LOG_LEVEL=INFO
TZ=UTC
```

## 🎮 **Gaming/Performance Mode (High-End GPU)**
For RTX 4090 or high-end setups:
```
OLLAMA_MAX_LOADED_MODELS=3
OLLAMA_NUM_PARALLEL=8
OLLAMA_MAX_QUEUE=1024
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
```

## 💰 **Budget Mode (Lower-End GPU)**
For RTX 3060/3070 or budget setups:
```
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_QUEUE=256
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

## 🔍 **Debug Mode**
For troubleshooting:
```
OLLAMA_DEBUG=1
ORCHESTRATOR_LOG_LEVEL=DEBUG
PYTHONUNBUFFERED=1
NPM_CONFIG_LOGLEVEL=verbose
```

## 📊 **Variable Explanations**

### **OLLAMA Configuration:**
- `OLLAMA_HOST`: Allow external connections
- `OLLAMA_ORIGINS`: Allow CORS from any origin
- `OLLAMA_MAX_LOADED_MODELS`: Number of models to keep in memory
- `OLLAMA_NUM_PARALLEL`: Parallel processing threads

### **Performance Optimization:**
- `PYTORCH_CUDA_ALLOC_CONF`: GPU memory management
- `CUDA_VISIBLE_DEVICES`: Which GPU to use
- `PIP_NO_CACHE_DIR`: Save disk space

### **Development Quality of Life:**
- `PYTHONUNBUFFERED`: Real-time Python output
- `NPM_CONFIG_PROGRESS`: Cleaner npm install output
- `DEBIAN_FRONTEND`: Non-interactive package installs

## 🚀 **Template Override Instructions**

1. **In RunPod Console**, when creating a pod
2. **Scroll down to "Template Override"**
3. **Click "Show Advanced Options"**
4. **Find "Environment Variables" section**
5. **Copy-paste the variables above**
6. **Deploy your pod**

## ⚙️ **Verification Commands**

After pod starts, verify environment variables:
```bash
# Check Python environment
echo $PYTHONPATH
echo $PYTHONUNBUFFERED

# Check Ollama settings
echo $OLLAMA_HOST
echo $OLLAMA_MAX_LOADED_MODELS

# Check GPU settings
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

# Check orchestrator settings
echo $ORCHESTRATOR_OUTPUT_DIR
echo $ORCHESTRATOR_LOG_LEVEL
```

## 🎯 **Recommended Minimal Set**

If you want to keep it simple, use these essential ones:
```
PYTHONUNBUFFERED=1
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_ORIGINS=*
NODE_ENV=development
DEBIAN_FRONTEND=noninteractive
ORCHESTRATOR_OUTPUT_DIR=/workspace/output
```