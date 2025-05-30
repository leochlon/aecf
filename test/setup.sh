#!/bin/bash

# AECF Ablation Suite Setup Script
# This script helps set up the environment for running the ablation suite

# Function to print colored output
print_step() {
    echo -e "\033[1;34m==>\033[0m \033[1m$1\033[0m"
}

print_error() {
    echo -e "\033[1;31mError:\033[0m \033[1m$1\033[0m"
}

print_success() {
    echo -e "\033[1;32mSuccess:\033[0m \033[1m$1\033[0m"
}

# Check Python version
print_step "Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
    print_error "Python 3.8 or higher required (found $PYTHON_VERSION)"
    exit 1
fi
print_success "Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
print_step "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
print_step "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -e .

# Verify installation
print_step "Verifying installation..."
VERIFY_CMD="from ablation.suite import AblationSuite; print('✓ Installation successful')"
if python3 -c "$VERIFY_CMD" &> /dev/null; then
    print_success "Ablation suite installed successfully!"
    echo "You can now run experiments with:"
    echo "  python cocoAblation.py --help"
else
    print_error "Installation verification failed"
    echo "Please check the error messages above"
    exit 1
fi
