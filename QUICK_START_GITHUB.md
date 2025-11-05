# Quick Start: Upload to GitHub

## Option 1: Automated Script (Easiest)

Run the automated upload script:

```bash
./upload_to_github.sh
```

This script will:
1. Check/configure git settings
2. Stage all files
3. Create initial commit
4. Create GitHub repository (if GitHub CLI installed) OR guide you through manual setup

## Option 2: Manual Setup (If GitHub CLI not available)

### Step 1: Create Repository on GitHub

1. Go to: https://github.com/new
2. Repository name: `quant-research-system`
3. Description: `AI-Driven Quantitative Trading Research System for Gold (XAU/USD) analysis`
4. Choose **Public** or **Private**
5. **IMPORTANT:** Do NOT check "Add a README file", "Add .gitignore", or "Choose a license" (we already have these)
6. Click **"Create repository"**

### Step 2: Copy Repository URL

After creating the repository, GitHub will show you the repository URL. Copy it.

Example: `https://github.com/YOUR_USERNAME/quant-research-system.git`

### Step 3: Run These Commands

```bash
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/quant-research-system.git

# Or if you prefer SSH (if you have SSH keys set up):
# git remote add origin git@github.com:YOUR_USERNAME/quant-research-system.git

# Stage all files
git add .

# Create initial commit
git commit -m "Initial commit: Complete quantitative trading research system

- Gold (XAU/USD) analysis from Jan 2022 - Oct 2025
- Multi-timeframe analysis (1min, 5min, 1hr, 4hr, daily)
- Technical indicator testing with entropy scoring
- Neural network regime classification (86.66% accuracy)
- Comprehensive research report with 14 visualizations
- Automated report generation and validation tools"

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 4: Verify Upload

1. Go to your repository on GitHub
2. Check that all files are present:
   - ✅ `README.md`
   - ✅ `reports/XAU_USD_Research_Report.md`
   - ✅ All `src/*.py` files
   - ✅ `requirements.txt`
   - ✅ `.gitignore`

## Troubleshooting

### Authentication Issues

If you get authentication errors:

```bash
# Use GitHub CLI for authentication
gh auth login

# Or set up personal access token:
# 1. Go to: https://github.com/settings/tokens
# 2. Generate new token (classic)
# 3. Use token as password when pushing
```

### Large Files

If you get errors about large files:

The `.gitignore` file excludes large data files by default. If you need to include them:

```bash
# Use Git LFS for large files
git lfs install
git lfs track "*.csv"
git lfs track "*.pth"
git add .gitattributes
```

### Repository Already Exists

If you get "repository already exists" error:

```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin https://github.com/YOUR_USERNAME/quant-research-system.git

# Push
git push -u origin main
```

## What Gets Uploaded

**Included:**
- ✅ All Python source files (`src/*.py`)
- ✅ README.md
- ✅ Research report (`reports/XAU_USD_Research_Report.md`)
- ✅ Configuration files (`config.py`, `requirements.txt`)
- ✅ Scripts (`scripts/*.py`)
- ✅ Documentation files
- ✅ `.gitignore` and `.github/` workflows

**Excluded (via .gitignore):**
- ❌ Large CSV data files (`data/raw/*.csv`, `data/processed/*.csv`)
- ❌ Model checkpoint files (`models/*.pth`)
- ❌ Python cache (`__pycache__/`)
- ❌ Virtual environment (`venv/`, `.venv/`)
- ❌ Environment variables (`.env`)
- ❌ IDE files (`.vscode/`, `.idea/`)

This keeps the repository size manageable while preserving all code and documentation.

