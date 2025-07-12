#!/bin/bash

# AI Development Team Orchestrator - RunPod Setup Script
# This script sets up the complete environment for running the AI orchestrator on RunPod

set -e  # Exit on any error

echo "🚀 Setting up AI Development Team Orchestrator for RunPod..."

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

# Check if running as root (RunPod containers run as root)
if [ "$EUID" -ne 0 ]; then
    print_warning "Not running as root, but continuing..."
fi

# Update system packages
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install essential dependencies
print_status "Installing essential dependencies..."
apt install -y curl wget git python3 python3-pip python3-venv build-essential

# Install Node.js for generated projects
print_status "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Verify installations
print_status "Verifying installations..."
python3 --version
node --version
npm --version
git --version

# Install Ollama
print_status "Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
print_status "Starting Ollama service..."
systemctl start ollama
systemctl enable ollama

# Wait for Ollama to start
print_status "Waiting for Ollama to start..."
sleep 10

# Verify Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null; then
    print_success "Ollama is running successfully!"
else
    print_error "Ollama failed to start. Please check the service."
    systemctl status ollama
    exit 1
fi

# Create Python virtual environment
print_status "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies if available
if [ -f "requirements-dev.txt" ]; then
    print_status "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Setup environment file
print_status "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp env.example .env
    print_success "Environment file created from template"
else
    print_warning "Environment file already exists, skipping creation"
fi

# Pull default model (Llama2 7B for testing)
print_status "Pulling default model (Llama2 7B)..."
print_warning "This will take 5-10 minutes depending on your connection..."
ollama pull llama2:7b

# Verify model is available
print_status "Verifying model availability..."
ollama list

# Test GPU setup
print_status "Testing GPU setup..."
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    print_warning "No NVIDIA GPU detected. Running in CPU mode."
fi

# Test the orchestrator setup
print_status "Testing orchestrator setup..."
if python test_gpu_setup.py; then
    print_success "Orchestrator setup test passed!"
else
    print_error "Orchestrator setup test failed. Please check the configuration."
    exit 1
fi

# Create useful aliases
print_status "Creating useful aliases..."
cat >> ~/.bashrc << 'EOF'

# AI Development Team Orchestrator Aliases
alias ai-orchestrator='cd /root/projectdelta && source venv/bin/activate && python main.py'
alias ai-status='systemctl status ollama && nvidia-smi'
alias ai-logs='tail -f orchestrator.log'
alias ai-restart='systemctl restart ollama'
EOF

# Print final status
echo ""
print_success "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit your environment: nano .env"
echo "2. Pull larger models: ollama pull llama2:70b"
echo "3. Start the orchestrator: python main.py"
echo ""
echo "🔧 Useful commands:"
echo "- Check GPU: nvidia-smi"
echo "- Monitor Ollama: journalctl -u ollama -f"
echo "- View logs: tail -f orchestrator.log"
echo "- Restart Ollama: systemctl restart ollama"
echo ""
echo "💡 Cost optimization:"
echo "- Use spot instances when available"
echo "- Stop pod when not in use"
echo "- Monitor GPU usage with nvidia-smi"
echo ""
print_success "Your AI Development Team Orchestrator is ready for RunPod! 🚀" 