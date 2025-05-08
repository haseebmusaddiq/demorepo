# Navigate to your project folder
cd "H:/final project/rag_service"

# Initialize Git repository
git init

# Add all files to staging
git add .

# Commit changes
git commit -m "Initial commit"

# Add remote repository
git remote add origin https://github.com/haseebmusaddiq/ConstitutionalRepository.git

# Pull existing content (to merge with your local repository)
git pull origin main --allow-unrelated-histories

# If there are merge conflicts, resolve them and commit:
# git add .
# git commit -m "Resolve merge conflicts"

# Push your code to GitHub
git push -u origin main