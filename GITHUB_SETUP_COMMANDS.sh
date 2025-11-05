#!/bin/bash
# GitHub Repository Setup Script
# Run this script after creating the repository on GitHub

echo "========================================="
echo "GITHUB REPOSITORY SETUP"
echo "========================================="
echo ""

# Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    echo "GitHub CLI detected. Creating repository..."
    read -p "Repository name (default: quant-research-system): " REPO_NAME
    REPO_NAME=${REPO_NAME:-quant-research-system}
    
    read -p "Make repository public? (y/n, default: y): " IS_PUBLIC
    IS_PUBLIC=${IS_PUBLIC:-y}
    
    if [ "$IS_PUBLIC" = "y" ]; then
        VISIBILITY="--public"
    else
        VISIBILITY="--private"
    fi
    
    echo "Creating repository: $REPO_NAME"
    gh repo create $REPO_NAME $VISIBILITY \
        --description "AI-Driven Quantitative Trading Research System for Gold (XAU/USD) analysis" \
        --source=. \
        --remote=origin \
        --push
    
    echo ""
    echo "✅ Repository created and pushed to GitHub!"
    echo "View at: https://github.com/$(gh api user --jq .login)/$REPO_NAME"
else
    echo "GitHub CLI not installed. Using manual setup..."
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://github.com/new"
    echo "2. Create repository: quant-research-system"
    echo "3. Run these commands:"
    echo ""
    echo "   git remote add origin https://github.com/YOUR_USERNAME/quant-research-system.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
fi
