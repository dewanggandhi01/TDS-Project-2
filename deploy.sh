#!/bin/bash

# Deployment script for the LLM Data Analyst Agent
echo "🚀 Deploying LLM Data Analyst Agent..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit"
fi

# Install Vercel CLI if not present
if ! command -v vercel &> /dev/null; then
    echo "Installing Vercel CLI..."
    npm install -g vercel
fi

# Deploy to Vercel
echo "Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo "📋 Remember to set these environment variables in Vercel dashboard:"
echo "   - OPENAI_API_KEY: Your OpenAI API key"
echo "   - ENVIRONMENT: production"
