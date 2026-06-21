#!/bin/bash
# Script to push soccer betting project to GitHub

echo "🚀 Pushing soccer-betting-models to GitHub..."
echo ""

# Remove any existing remote
git remote remove origin 2>/dev/null || true

# Add the GitHub remote
git remote add origin https://github.com/pabsanamono/soccer-betting-models.git

# Rename branch to main (GitHub's default)
git branch -M main

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Success! Your repository is now live at:"
    echo "   https://github.com/pabsanamono/soccer-betting-models"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Visit your repository and add topics/tags"
    echo "   2. Consider adding a license"
    echo "   3. Star your own repo! ⭐"
else
    echo ""
    echo "❌ Push failed. Make sure you created the repository on GitHub first:"
    echo "   https://github.com/new"
fi
