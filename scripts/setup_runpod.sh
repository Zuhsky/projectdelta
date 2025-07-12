#!/bin/bash

# AI Development Team Orchestrator - RunPod GPU Setup Script
# This script sets up the environment for high-performance GPU usage

set -e  # Exit on any error

echo "🚀 Setting up AI Development Team Orchestrator for RunPod GPU..."
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is installed
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.8+ is required, found $PYTHON_VERSION"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python"
        else
            print_error "Python 3.8+ is required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python is not installed. Please install Python 3.8+ and try again."
        exit 1
    fi
}

# Check GPU availability
check_gpu() {
    print_status "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        if [ $? -eq 0 ]; then
            print_success "GPU detected: $GPU_INFO"
            
            # Check GPU memory
            GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
            if [ "$GPU_MEMORY" -ge 12000 ]; then
                print_success "GPU memory: ${GPU_MEMORY}MB (sufficient for 30B+ models)"
            else
                print_warning "GPU memory: ${GPU_MEMORY}MB (may be limited for 30B+ models)"
            fi
        else
            print_error "nvidia-smi failed to get GPU information"
            exit 1
        fi
    else
        print_error "nvidia-smi not found. GPU may not be available."
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Update package list
    apt update
    
    # Install essential packages
    apt install -y \
        curl wget git build-essential \
        python3-pip python3-dev \
        nvidia-cuda-toolkit \
        htop nvtop \
        tmux screen
    
    print_success "System dependencies installed"
}

# Install Node.js for generated projects
install_nodejs() {
    print_status "Installing Node.js..."
    
    if ! command -v node &> /dev/null; then
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        apt install -y nodejs
        
        NODE_VERSION=$(node --version)
        NPM_VERSION=$(npm --version)
        print_success "Node.js $NODE_VERSION and npm $NPM_VERSION installed"
    else
        print_success "Node.js already installed"
    fi
}

# Install Ollama
install_ollama() {
    print_status "Installing Ollama..."
    
    if ! command -v ollama &> /dev/null; then
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Add ollama to PATH
        export PATH=$PATH:/usr/local/bin
        
        print_success "Ollama installed"
    else
        print_success "Ollama already installed"
    fi
    
    # Start Ollama with GPU optimization
    print_status "Starting Ollama with GPU optimization..."
    
    # Kill existing ollama processes
    pkill -f "ollama serve" || true
    
    # Start Ollama with optimized settings
    nohup ollama serve > /workspace/ollama.log 2>&1 &
    
    # Wait for service to start
    sleep 10
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama service started successfully"
    else
        print_error "Failed to start Ollama service"
        exit 1
    fi
}

# Pull 30B+ models
pull_models() {
    print_status "Pulling 30B+ models for GPU acceleration..."
    
    models=(
        "llama2:70b-chat"
        "deepseek-coder:33b"
        "codellama:70b-instruct"
        "mixtral:8x7b-instruct"
    )
    
    for model in "${models[@]}"; do
        print_status "Pulling model: $model"
        
        # Check if model already exists
        if ollama list | grep -q "$model"; then
            print_success "Model $model already exists"
        else
            if ollama pull "$model"; then
                print_success "Model $model pulled successfully"
            else
                print_error "Failed to pull model: $model"
                print_warning "You can pull it manually later with: ollama pull $model"
            fi
        fi
    done
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        $PYTHON_CMD -m pip install --upgrade pip
        $PYTHON_CMD -m pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Setup environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cp env.example .env
        print_success "Created .env file from template"
    else
        print_success ".env file already exists"
    fi
    
    # Set GPU environment variables
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
    export OLLAMA_HOST=0.0.0.0:11434
    export OLLAMA_ORIGINS=*
    export OLLAMA_MAX_LOADED_MODELS=3
    export OLLAMA_NUM_PARALLEL=8
    export OLLAMA_MAX_QUEUE=1024
    
    print_success "Environment variables configured"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=(
        "/workspace/output"
        "/workspace/data"
        "/workspace/logs"
        "/workspace/models"
        "/workspace/projects"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_success "Created directory: $dir"
    done
}

# Test GPU setup
test_gpu_setup() {
    print_status "Testing GPU setup..."
    
    # Test nvidia-smi
    if nvidia-smi > /dev/null 2>&1; then
        print_success "nvidia-smi working correctly"
    else
        print_error "nvidia-smi test failed"
        exit 1
    fi
    
    # Test Ollama with GPU
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama API accessible"
    else
        print_error "Ollama API test failed"
        exit 1
    fi
    
    # Test Python GPU libraries
    if $PYTHON_CMD -c "import GPUtil; print('GPUtil working')" 2>/dev/null; then
        print_success "Python GPU libraries working"
    else
        print_warning "Python GPU libraries test failed"
    fi
}

# Display completion message
display_completion() {
    echo
    echo "======================================"
    print_success "RunPod GPU setup completed successfully!"
    echo "======================================"
    echo
    echo "🎉 Your GPU-optimized AI Development Team Orchestrator is ready!"
    echo
    echo "🖥️ GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,temperature.gpu --format=csv,noheader
    echo
    echo "📦 Installed Models:"
    ollama list
    echo
    echo "🚀 To get started:"
    echo "  $PYTHON_CMD main.py"
    echo
    echo "📊 Monitor GPU usage:"
    echo "  nvidia-smi"
    echo "  nvtop"
    echo
    echo "📝 View logs:"
    echo "  tail -f /workspace/ollama.log"
    echo "  tail -f /workspace/logs/orchestrator_*.log"
    echo
    print_success "Happy coding with your GPU-accelerated AI development team! 🚀"
}

# Main setup function
main() {
    echo "AI Development Team Orchestrator - RunPod GPU Setup"
    echo "=================================================="
    echo
    
    # Run all checks and installations
    check_python
    check_gpu
    install_system_deps
    install_nodejs
    install_ollama
    pull_models
    install_python_deps
    setup_environment
    create_directories
    test_gpu_setup
    display_completion
}

# Run main function
main "$@" 