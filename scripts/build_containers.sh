#!/bin/bash
# =============================================================================
# CASC Docker Image Build Script
# =============================================================================
# This script builds and optionally pushes Docker images for the CASC pipeline.
#
# Usage:
#   ./build_containers.sh [model] [--push]
#
# Examples:
#   ./build_containers.sh                 # Build all images
#   ./build_containers.sh cinema          # Build only CineMA image
#   ./build_containers.sh all --push      # Build and push all images
#   ./build_containers.sh nnformer --push # Build and push nnFormer image
# =============================================================================

set -e  # Exit on error

# Configuration
REGISTRY="ghcr.io"
NAMESPACE="pmoheban"
VERSION="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory (CASC root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$SCRIPT_DIR"

# Models and their Dockerfiles
declare -A MODELS
MODELS=(
    ["cinema"]="containers/cinema/Dockerfile"
    ["nnformer"]="containers/nnformer/Dockerfile"
    ["vsa3l"]="containers/vsa3l/Dockerfile"
)

# Functions
print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        echo "See: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker is installed: $(docker --version)"
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    print_success "Docker daemon is running"
    
    # Check GPU support (optional)
    if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU support is available"
    else
        print_warning "NVIDIA GPU support not available (images will still build)"
    fi
}

check_model_files() {
    local model=$1
    print_header "Checking files for $model"
    
    # Check Dockerfile
    if [[ ! -f "${MODELS[$model]}" ]]; then
        print_error "Dockerfile not found: ${MODELS[$model]}"
        return 1
    fi
    print_success "Dockerfile found: ${MODELS[$model]}"
    
    # Check bin scripts exist
    if [[ -d "bin" ]]; then
        print_success "bin/ directory found"
    else
        print_warning "bin/ directory not found - scripts may be missing"
    fi
    
    # Check model weights (warn if missing)
    case $model in
        cinema)
            if [[ -f "models/cinema/model.safetensors" ]]; then
                print_success "CineMA model weights found"
            else
                print_warning "CineMA model weights not found at models/cinema/model.safetensors"
                print_warning "Image will build but you'll need to add weights later"
            fi
            ;;
        nnformer)
            if [[ -f "models/nnformer/model.pkl" ]] || ls models/nnformer/*.pkl &> /dev/null; then
                print_success "nnFormer model weights found"
            else
                print_warning "nnFormer model weights not found at models/nnformer/"
                print_warning "Image will build but you'll need to add weights later"
            fi
            ;;
        vsa3l)
            if [[ -f "models/vsa3l/model.pt" ]]; then
                print_success "VSA-3L model weights found"
            else
                print_warning "VSA-3L model weights not found at models/vsa3l/model.pt"
                print_warning "Image will build but you'll need to add weights later"
            fi
            ;;
    esac
    
    return 0
}

build_image() {
    local model=$1
    local image_name="${REGISTRY}/${NAMESPACE}/casc-${model}:${VERSION}"
    
    print_header "Building $model image: $image_name"
    
    # Check files first
    if ! check_model_files "$model"; then
        print_error "Skipping $model due to missing files"
        return 1
    fi
    
    echo -e "\nBuilding image (this may take a while)...\n"
    
    # Build command
    if docker build \
        -f "${MODELS[$model]}" \
        -t "$image_name" \
        --progress=plain \
        .; then
        print_success "Successfully built: $image_name"
        echo ""
        docker images "$image_name"
        return 0
    else
        print_error "Failed to build: $image_name"
        return 1
    fi
}

push_image() {
    local model=$1
    local image_name="${REGISTRY}/${NAMESPACE}/casc-${model}:${VERSION}"
    
    print_header "Pushing $model image: $image_name"
    
    # Check if logged in
    if ! docker push "$image_name" 2>&1 | head -5; then
        print_error "Push failed. Make sure you're logged in:"
        echo ""
        echo "  echo \$GITHUB_TOKEN | docker login ghcr.io -u \$GITHUB_USER --password-stdin"
        echo ""
        return 1
    fi
    
    print_success "Successfully pushed: $image_name"
    return 0
}

show_usage() {
    echo "CASC Docker Build Script"
    echo ""
    echo "Usage: $0 [model] [--push]"
    echo ""
    echo "Models:"
    echo "  all       Build all models (default)"
    echo "  cinema    Build CineMA model only"
    echo "  nnformer  Build nnFormer model only"
    echo "  vsa3l     Build VSA-3L model only"
    echo ""
    echo "Options:"
    echo "  --push    Push images to registry after building"
    echo "  --help    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Build all images"
    echo "  $0 cinema              # Build CineMA only"
    echo "  $0 all --push          # Build and push all"
    echo "  $0 nnformer --push     # Build and push nnFormer"
    echo ""
}

# =============================================================================
# Main Script
# =============================================================================

MODEL_ARG=""
PUSH_ARG=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --push)
            PUSH_ARG=true
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        cinema|nnformer|vsa3l|all)
            MODEL_ARG=$arg
            ;;
        *)
            print_error "Unknown argument: $arg"
            show_usage
            exit 1
            ;;
    esac
done

# Default to all
if [[ -z "$MODEL_ARG" ]]; then
    MODEL_ARG="all"
fi

# Check prerequisites
check_prerequisites

# Determine which models to build
if [[ "$MODEL_ARG" == "all" ]]; then
    MODELS_TO_BUILD=("cinema" "nnformer" "vsa3l")
else
    MODELS_TO_BUILD=("$MODEL_ARG")
fi

# Build images
print_header "Starting Build Process"
echo "Models to build: ${MODELS_TO_BUILD[*]}"
echo "Push after build: $PUSH_ARG"

BUILD_SUCCESS=()
BUILD_FAILED=()

for model in "${MODELS_TO_BUILD[@]}"; do
    if build_image "$model"; then
        BUILD_SUCCESS+=("$model")
        
        if $PUSH_ARG; then
            if ! push_image "$model"; then
                print_warning "Push failed for $model"
            fi
        fi
    else
        BUILD_FAILED+=("$model")
    fi
done

# Summary
print_header "Build Summary"

if [[ ${#BUILD_SUCCESS[@]} -gt 0 ]]; then
    print_success "Successfully built: ${BUILD_SUCCESS[*]}"
fi

if [[ ${#BUILD_FAILED[@]} -gt 0 ]]; then
    print_error "Failed to build: ${BUILD_FAILED[*]}"
fi

echo ""
echo "Built images:"
docker images | grep "casc-"
echo ""

if ! $PUSH_ARG; then
    echo "To push images to the registry, run:"
    echo "  $0 ${MODEL_ARG} --push"
    echo ""
fi

echo "Done!"
