# Use official Go image as build environment
FROM golang:1.22-alpine AS builder

# Set working directory
WORKDIR /app

# Copy go.mod and go.sum
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build application
RUN CGO_ENABLED=0 GOOS=linux go build -o main .

# Use lightweight alpine as runtime environment
FROM alpine:latest

# Install ca-certificates
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy binary from build stage
COPY --from=builder /app/main .

# Set environment variables
ENV PORT=8080
ENV ENABLE_PROXY=false
ENV PROXY_URL=""
ENV PROXY_TIMEOUT_MS=5000
ENV LOG_LEVEL=info

# Expose port
EXPOSE 8080

# Run the application
CMD ["./main"]