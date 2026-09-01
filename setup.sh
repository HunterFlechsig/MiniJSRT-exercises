set -e

VENV_DIR="venv"

# Create virtual environment if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements if available
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

echo ""
echo "Setup complete!"
echo "Virtual environment is active."
echo "Python: $(which python)"
echo "Pip:    $(which pip)"