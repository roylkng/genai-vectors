#!/bin/bash
set -e

echo "🔹 Unsetting GCC environment..."
unset CC
unset CXX

echo "🔹 Installing LLVM + libomp if missing..."
brew install llvm libomp

echo "🔹 Cleaning Cargo + FAISS build cache..."
cargo clean
rm -rf ~/.cargo/registry/src/index.crates.io-*/faiss-sys-*
rm -rf target

echo "🔹 Setting Clang + libomp build environment..."
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
export LDFLAGS="-L$(brew --prefix libomp)/lib"
export CPPFLAGS="-I$(brew --prefix libomp)/include"
export LIBRARY_PATH="$(brew --prefix libomp)/lib"
export CMAKE_ARGS="-DFAISS_ENABLE_OPENMP=ON \
  -DOpenMP_C_FLAGS=-fopenmp \
  -DOpenMP_CXX_FLAGS=-fopenmp \
  -DOpenMP_C_LIB_NAMES=omp \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib"

echo "🔹 Rebuilding with Cargo..."
cargo build

echo "✅ Build completed without gomp errors!"

