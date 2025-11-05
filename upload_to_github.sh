#!/bin/bash
# Script to upload project to GitHub
# This script will help you create a new GitHub repository and upload all files

echo "========================================="
echo "GITHUB REPOSITORY UPLOAD SCRIPT"
echo "========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Check git config
if [ -z "$(git config user.name)" ]; then
    echo "⚠️  Git user name not configured"
    read -p "Enter your Git user name: " GIT_USER
    git config user.name "$GIT_USER"
fi

if [ -z "$(git config user.email)" ]; then
    echo "⚠️  Git email not configured"
    read -p "Enter your Git email: " GIT_EMAIL
    git config user.email "$GIT_EMAIL"
fi

echo ""
echo "Git Configuration:"
echo "  User: $(git config user.name)"
echo "  Email: $(git config user.email)"
echo ""

# Check if GitHub CLI is available
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI detected!"
    echo ""
    read -p "Repository name (default: quant-research-system): " REPO_NAME
    REPO_NAME=${REPO_NAME:-quant-research-system}
    
    read -p "Make repository public? (y/n, default: y): " IS_PUBLIC
    IS_PUBLIC=${IS_PUBLIC:-y}
    
    if [ "$IS_PUBLIC" = "y" ]; then
        VISIBILITY="--public"
    else
        VISIBILITY="--private"
    fi
    
    echo ""
    echo "Creating repository on GitHub..."
    echo "  Name: $REPO_NAME"
    echo "  Visibility: $VISIBILITY"
    echo ""
    
    # Stage all files
    echo "Staging all files..."
    git add .
    
    # Create initial commit
    echo "Creating initial commit..."
    git commit -m "Initial commit: Complete quantitative trading research system

- Gold (XAU/USD) analysis from Jan 2022 - Oct 2025
- Multi-timeframe analysis (1min, 5min, 1hr, 4hr, daily)
- Technical indicator testing with entropy scoring
- Neural network regime classification (86.66% accuracy)
- Comprehensive research report with 14 visualizations
- Automated report generation and validation tools"
    
    # Create repository and push
    echo ""
    echo "Creating repository on GitHub and pushing code..."
    gh repo create $REPO_NAME $VISIBILITY \
        --description "AI-Driven Quantitative Trading Research System for Gold (XAU/USD) analysis" \
        --source=. \
        --remote=origin \
        --push
    
    echo ""
    echo "✅ Repository created and pushed successfully!"
    echo ""
    REPO_URL=$(gh repo view $REPO_NAME --json url -q .url)
    echo "View your repository at: $REPO_URL"
    
else
    echo "⚠️  GitHub CLI not installed. Using manual method..."
    echo ""
    echo "Please follow these steps:"
    echo ""
    echo "1. Create repository on GitHub:"
    echo "   Go to: https://github.com/new"
    echo "   Repository name: quant-research-system"
    echo "   Description: AI-Driven Quantitative Trading Research System for Gold (XAU/USD) analysis"
    echo "   Choose Public or Private"
    echo "   DO NOT initialize with README, .gitignore, or license"
    echo "   Click 'Create repository'"
    echo ""
    echo "2. After creating the repository, run these commands:"
    echo ""
    echo "   # Add remote (replace YOUR_USERNAME with your GitHub username)"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/quant-research-system.git"
    echo ""
    echo "   # Or if using SSH:"
    echo "   git remote add origin git@github.com:YOUR_USERNAME/quant-research-system.git"
    echo ""
    echo "   # Stage all files"
    echo "   git add ."
    echo ""
    echo "   # Create commit"
    echo "   git commit -m 'Initial commit: Complete quantitative trading research system'"
    echo ""
    echo "   # Push to GitHub"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo ""
    echo "Would you like me to stage all files and prepare for commit now? (y/n)"
    read -p "> " PREPARE_NOW
    
    if [ "$PREPARE_NOW" = "y" ]; then
        echo ""
        echo "Staging all files..."
        git add .
        
        echo ""
        echo "Files staged. Creating commit..."
        git commit -m "Initial commit: Complete quantitative trading research system

- Gold (XAU/USD) analysis from Jan 2022 - Oct 2025
- Multi-timeframe analysis (1min, 5min, 1hr, 4hr, daily)
- Technical indicator testing with entropy scoring
- Neural network regime classification (86.66% accuracy)
- Comprehensive research report with 14 visualizations
- Automated report generation and validation tools"
        
        echo ""
        echo "✅ Commit created successfully!"
        echo ""
        echo "Next steps:"
        echo "1. Create repository on GitHub: https://github.com/new"
        echo "2. Add remote: git remote add origin https://github.com/YOUR_USERNAME/quant-research-system.git"
        echo "3. Push: git branch -M main && git push -u origin main"
    fi
fi

echo ""
echo "========================================="
echo "SETUP COMPLETE"
echo "========================================="

