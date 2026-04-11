#!/bin/bash
# SigmaTau repo setup — run this once to create the directory structure
# Usage: bash setup.sh /path/to/legacy/matlab /path/to/StabLab.jl /path/to/KalmanLab.jl

set -e

LEGACY_MATLAB="${1:-}"
LEGACY_STABLAB="${2:-}"
LEGACY_KALMANLAB="${3:-}"

echo "=== Setting up SigmaTau ==="

# ── MATLAB package structure ──
mkdir -p matlab/+sigmatau/{+dev,+noise,+stats,+kf,+steering,+plot,+util}
mkdir -p matlab/{examples,tests,legacy}

# ── Julia package structure ──
mkdir -p julia/{src,test,examples}
mkdir -p julia/ext

# ── Shared docs ──
mkdir -p docs

# ── Copy legacy code if paths provided ──
if [ -n "$LEGACY_MATLAB" ] && [ -d "$LEGACY_MATLAB" ]; then
    echo "Copying legacy MATLAB code to matlab/legacy/"
    cp "$LEGACY_MATLAB"/*.m matlab/legacy/ 2>/dev/null || true
    echo "  Copied $(ls matlab/legacy/*.m 2>/dev/null | wc -l) files"
else
    echo "SKIP: No legacy MATLAB path provided (arg 1)"
    echo "  Usage: bash setup.sh /path/to/AllanLab /path/to/StabLab.jl /path/to/KalmanLab.jl"
fi

if [ -n "$LEGACY_STABLAB" ] && [ -d "$LEGACY_STABLAB" ]; then
    mkdir -p julia/legacy_stablab
    echo "Copying legacy StabLab.jl code to julia/legacy_stablab/"
    cp -r "$LEGACY_STABLAB"/src/* julia/legacy_stablab/ 2>/dev/null || true
fi

if [ -n "$LEGACY_KALMANLAB" ] && [ -d "$LEGACY_KALMANLAB" ]; then
    mkdir -p julia/legacy_kalmanlab
    echo "Copying legacy KalmanLab.jl code to julia/legacy_kalmanlab/"
    cp -r "$LEGACY_KALMANLAB"/src/* julia/legacy_kalmanlab/ 2>/dev/null || true
fi

# ── Git init ──
if [ ! -d ".git" ]; then
    git init
    cat > .gitignore << 'EOF'
# MATLAB
*.asv
*.mex*
*.mlappinstall
*.mltbx

# Julia
julia/Manifest.toml
julia/.CondaPkg/

# OS
.DS_Store
Thumbs.db

# Editor
*.swp
*.swo
*~
.vscode/
.idea/

# Results (generated, not tracked)
results/
steering_results/
EOF
    git add -A
    git commit -m "init: SigmaTau repo structure with legacy code"
    echo ""
    echo "=== Git initialized and first commit made ==="
fi

echo ""
echo "=== SigmaTau repo ready ==="
echo ""
echo "Directory structure:"
find . -type d -not -path './.git/*' -not -path './.git' | sort | head -40
echo ""
echo "Next steps:"
echo "  1. cd into this directory"
echo "  2. Run: claude"
echo "  3. First prompt: 'Read CLAUDE.md. Start with the deviation engine.'"
echo "     Or use: /implement engine"
