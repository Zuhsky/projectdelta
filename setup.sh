#!/bin/bash

# AI Development Team Orchestrator Setup Script
# This script sets up the environment and installs all necessary dependencies

set -e  # Exit on any error

echo "🚀 Setting up AI Development Team Orchestrator..."
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
        echo "Visit https://python.org for installation instructions."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        print_success "pip found"
        PIP_CMD="pip"
    else
        print_error "pip is not installed. Please install pip and try again."
        exit 1
    fi
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        $PIP_CMD install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Check if Ollama is installed
check_ollama() {
    print_status "Checking Ollama installation..."
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama found"
        
        # Check if Ollama service is running
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_success "Ollama service is running"
        else
            print_warning "Ollama is installed but not running"
            print_status "Starting Ollama service..."
            
            # Try to start Ollama in the background
            if command -v systemctl &> /dev/null; then
                sudo systemctl start ollama 2>/dev/null || true
            fi
            
            # Start Ollama in background if not already running
            if ! pgrep -f "ollama serve" > /dev/null; then
                print_status "Starting Ollama server..."
                nohup ollama serve > /dev/null 2>&1 &
                sleep 3
            fi
            
            # Check again
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                print_success "Ollama service started"
            else
                print_error "Could not start Ollama service"
                echo "Please start Ollama manually:"
                echo "  ollama serve"
                exit 1
            fi
        fi
    else
        print_error "Ollama is not installed"
        echo
        echo "Please install Ollama:"
        echo "  • Linux/WSL: curl -fsSL https://ollama.ai/install.sh | sh"
        echo "  • macOS: brew install ollama"
        echo "  • Windows: Download from https://ollama.ai"
        echo
        echo "After installation, run this setup script again."
        exit 1
    fi
}

# Pull required Ollama models
pull_models() {
    print_status "Pulling required Ollama models..."
    
    models=("llama2:7b-chat" "deepseek-coder:33b")
    
    for model in "${models[@]}"; do
        print_status "Pulling model: $model"
        if ollama pull "$model"; then
            print_success "Model $model pulled successfully"
        else
            print_error "Failed to pull model: $model"
            echo "You can pull it manually later with: ollama pull $model"
        fi
    done
}

# Check if Node.js is installed (for generated projects)
check_nodejs() {
    print_status "Checking Node.js installation..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js $NODE_VERSION found"
    else
        print_warning "Node.js is not installed"
        echo "Node.js is required to run the generated Next.js applications."
        echo "Install Node.js from https://nodejs.org (recommended: LTS version)"
    fi
    
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        print_success "npm $NPM_VERSION found"
    else
        print_warning "npm is not installed"
    fi
}

# Check if Git is installed
check_git() {
    print_status "Checking Git installation..."
    
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version)
        print_success "$GIT_VERSION found"
    else
        print_warning "Git is not installed"
        echo "Git is required for repository management."
        echo "Install Git from https://git-scm.com"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=("data" "output")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        fi
    done
}

# Make main.py executable
make_executable() {
    print_status "Making main.py executable..."
    chmod +x main.py
    print_success "main.py is now executable"
}

# Display completion message
display_completion() {
    echo
    echo "======================================"
    print_success "Setup completed successfully!"
    echo "======================================"
    echo
    echo "🎉 Your AI Development Team Orchestrator is ready!"
    echo
    echo "To get started:"
    echo "  $PYTHON_CMD main.py"
    echo
    echo "Or run the interactive wizard:"
    echo "  $PYTHON_CMD main.py --help"
    echo
    echo "For troubleshooting:"
    echo "  • Ensure Ollama is running: ollama serve"
    echo "  • Check model availability: ollama list"
    echo "  • View logs in the terminal output"
    echo
    print_success "Happy coding with your AI development team! 🚀"
}

# Main setup function
main() {
    echo "AI Development Team Orchestrator Setup"
    echo "======================================"
    echo
    
    # Run all checks and installations
    check_python
    check_pip
    install_python_deps
    check_ollama
    pull_models
    check_nodejs
    check_git
    create_directories
    make_executable
    
    # Display completion message
    display_completion
}

# Run main function
main "$@"