FROM node:18-alpine

WORKDIR /app

# Install pnpm and dependencies
COPY package.json pnpm-lock.yaml* ./
RUN npm install -g pnpm@12 && pnpm install

# Copy app source
COPY . .

# Build args for environment
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

EXPOSE 3000

# Development mode by default (docker-compose overrides with pnpm dev)
CMD ["pnpm", "run", "dev"]
