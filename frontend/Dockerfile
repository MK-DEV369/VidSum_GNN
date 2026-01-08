FROM node:18-alpine

WORKDIR /app

# Copy package files first (for better layer caching)
COPY package.json package-lock.json* ./

# Install dependencies with npm ci for reproducible builds
RUN npm ci --verbose

# Copy project files
COPY . .

# Verify dependencies are installed
RUN npm list gsap motion three 2>/dev/null || echo "Warning: Some optional dependencies may not be available"

EXPOSE 5173

# Run with host binding for Docker
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
