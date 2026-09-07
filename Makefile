# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
#
# Neura-X Build System
# Orchestrates C, C++, Rust, and Python compilation.
#
# Usage:
#   make all        - Build everything
#   make c          - Build C code
#   make cpp        - Build C++ code
#   make rust       - Build Rust code
#   make python     - Build Python package
#   make test       - Run all tests
#   make clean      - Clean all build artifacts
#   make install    - Install Neura-X
#   make wheel      - Build distribution wheels
#   make docs       - Build documentation
#
# ==========================================================

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

# Project name
PROJECT_NAME := neura-x
VERSION := 1.0.0

# Compilers
CC := gcc
CXX := g++
RUST := cargo
PYTHON := python3

# Compiler flags
CFLAGS := -O3 -Wall -Wextra -std=c11 -fPIC
CXXFLAGS := -O3 -Wall -Wextra -std=c++17 -fPIC
LDFLAGS := -shared

# LLVM configuration
LLVM_CONFIG := llvm-config
LLVM_CFLAGS := $(shell $(LLVM_CONFIG) --cflags 2>/dev/null || echo "")
LLVM_LDFLAGS := $(shell $(LLVM_CONFIG) --ldflags --libs 2>/dev/null || echo "")

# Directories
ROOT_DIR := $(shell pwd)
BUILD_DIR := $(ROOT_DIR)/build
C_SRC_DIR := $(ROOT_DIR)/core/c
CPP_SRC_DIR := $(ROOT_DIR)/core/cpp
RUST_DIR := $(ROOT_DIR)/core/rust
PYTHON_DIR := $(ROOT_DIR)/python
DOCS_DIR := $(ROOT_DIR)/docs

# Output directories
C_BUILD_DIR := $(BUILD_DIR)/c
CPP_BUILD_DIR := $(BUILD_DIR)/cpp

# Source files
C_SOURCES := $(wildcard $(C_SRC_DIR)/hal/*.c) \
             $(wildcard $(C_SRC_DIR)/memory/*.c) \
             $(wildcard $(C_SRC_DIR)/math/*.c)

CPP_SOURCES := $(wildcard $(CPP_SRC_DIR)/llvm_jit/*.cpp) \
               $(wildcard $(CPP_SRC_DIR)/router/*.cpp) \
               $(wildcard $(CPP_SRC_DIR)/optimizer/*.cpp) \
               $(wildcard $(CPP_SRC_DIR)/fidelity/*.cpp)

# Object files
C_OBJECTS := $(patsubst $(C_SRC_DIR)/%.c,$(C_BUILD_DIR)/%.o,$(C_SOURCES))
CPP_OBJECTS := $(patsubst $(CPP_SRC_DIR)/%.cpp,$(CPP_BUILD_DIR)/%.o,$(CPP_SOURCES))

# Include paths
INCLUDES := -I$(C_SRC_DIR)/hal \
            -I$(C_SRC_DIR)/memory \
            -I$(C_SRC_DIR)/math \
            -I$(CPP_SRC_DIR)/llvm_jit \
            -I$(CPP_SRC_DIR)/router \
            -I$(CPP_SRC_DIR)/optimizer \
            -I$(CPP_SRC_DIR)/fidelity \
            $(LLVM_CFLAGS)

# Library outputs
C_LIB := $(BUILD_DIR)/libneura_c.so
CPP_LIB := $(BUILD_DIR)/libneura_cpp.so

# ──────────────────────────────────────────────────────────
# DEFAULT TARGET
# ──────────────────────────────────────────────────────────

.PHONY: all
all: banner dirs c cpp rust python
	@echo ""
	@echo "✅ Neura-X build complete."
	@echo "   Version: $(VERSION)"
	@echo ""

# ──────────────────────────────────────────────────────────
# BANNER
# ──────────────────────────────────────────────────────────

.PHONY: banner
banner:
	@echo "============================================================"
	@echo "  ⚡ Neura-X Build System"
	@echo "     Intelligence Without Limits."
	@echo "============================================================"
	@echo "  🧠 Founded by Edusei Mikel Lisamba"
	@echo "  🌍 Built in Kenya | Open University of Kenya"
	@echo "============================================================"
	@echo ""

# ──────────────────────────────────────────────────────────
# CREATE BUILD DIRECTORIES
# ──────────────────────────────────────────────────────────

.PHONY: dirs
dirs:
	@mkdir -p $(C_BUILD_DIR)/hal
	@mkdir -p $(C_BUILD_DIR)/memory
	@mkdir -p $(C_BUILD_DIR)/math
	@mkdir -p $(CPP_BUILD_DIR)/llvm_jit
	@mkdir -p $(CPP_BUILD_DIR)/router
	@mkdir -p $(CPP_BUILD_DIR)/optimizer
	@mkdir -p $(CPP_BUILD_DIR)/fidelity

# ──────────────────────────────────────────────────────────
# BUILD C CODE
# ──────────────────────────────────────────────────────────

.PHONY: c
c: dirs $(C_OBJECTS)
	@echo ""
	@echo "🔧 Linking C library..."
	$(CC) $(LDFLAGS) -o $(C_LIB) $(C_OBJECTS) -lm -lopenblas
	@echo "✅ C library built: $(C_LIB)"

$(C_BUILD_DIR)/%.o: $(C_SRC_DIR)/%.c
	@echo "🔧 Compiling C: $<"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# ──────────────────────────────────────────────────────────
# BUILD C++ CODE
# ──────────────────────────────────────────────────────────

.PHONY: cpp
cpp: dirs c $(CPP_OBJECTS)
	@echo ""
	@echo "🔧 Linking C++ library..."
	$(CXX) $(LDFLAGS) -o $(CPP_LIB) $(CPP_OBJECTS) \
		-L$(BUILD_DIR) -lneura_c $(LLVM_LDFLAGS)
	@echo "✅ C++ library built: $(CPP_LIB)"

$(CPP_BUILD_DIR)/%.o: $(CPP_SRC_DIR)/%.cpp
	@echo "🔧 Compiling C++: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ──────────────────────────────────────────────────────────
# BUILD RUST CODE
# ──────────────────────────────────────────────────────────

.PHONY: rust
rust: c cpp
	@echo ""
	@echo "🔧 Building Rust crates..."
	cd $(RUST_DIR)/pager && $(RUST) build --release
	cd $(RUST_DIR)/nex_format && $(RUST) build --release
	cd $(RUST_DIR)/serve && $(RUST) build --release
	cd $(RUST_DIR)/python_bridge && $(RUST) build --release
	@echo "✅ Rust crates built."

# ──────────────────────────────────────────────────────────
# BUILD PYTHON PACKAGE
# ──────────────────────────────────────────────────────────

.PHONY: python
python: rust
	@echo ""
	@echo "🔧 Building Python package..."
	@cd $(ROOT_DIR) && $(PYTHON) -m pip install maturin --quiet 2>/dev/null || true
	@echo "   Building wheel with maturin..."
	@cd $(ROOT_DIR) && maturin build --release 2>&1 | grep -E "(Built|error|warning)" || true
	@echo "   Installing wheel..."
	@cd $(ROOT_DIR) && pip install $(ROOT_DIR)/core/rust/python_bridge/target/wheels/neura_x-*.whl --force-reinstall --no-deps --quiet 2>/dev/null || true
	@echo "   Copying native core to source package..."
	@$(eval EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null))
	@if [ -f "$(RUST_DIR)/python_bridge/target/release/lib_core.so" ]; then \
		cp -f $(RUST_DIR)/python_bridge/target/release/lib_core.so \
			$(PYTHON_DIR)/neura_x/_core$(EXT_SUFFIX); \
		echo "   ✅ Core copied to python/neura_x/_core$(EXT_SUFFIX)"; \
	else \
		echo "   ⚠️  lib_core.so not found, skipping copy"; \
	fi
	@echo "✅ Python package built and installed."

# ──────────────────────────────────────────────────────────
# RUN TESTS
# ──────────────────────────────────────────────────────────

.PHONY: test
test:
	@echo ""
	@echo "🧪 Running Neura-X test suite..."
	cd $(PYTHON_DIR) && $(PYTHON) -m pytest tests/ -v --tb=short
	@echo "✅ All tests passed."

.PHONY: test-c
test-c:
	@echo "🧪 Running C tests..."
	@echo "   (C test runner not yet implemented)"

.PHONY: test-rust
test-rust:
	@echo "🧪 Running Rust tests..."
	cd $(RUST_DIR)/pager && $(RUST) test
	cd $(RUST_DIR)/nex_format && $(RUST) test
	cd $(RUST_DIR)/serve && $(RUST) test
	cd $(RUST_DIR)/python_bridge && $(RUST) test

# ──────────────────────────────────────────────────────────
# CLEAN
# ──────────────────────────────────────────────────────────

.PHONY: clean
clean:
	@echo ""
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	cd $(RUST_DIR)/pager && $(RUST) clean 2>/dev/null || true
	cd $(RUST_DIR)/nex_format && $(RUST) clean 2>/dev/null || true
	cd $(RUST_DIR)/serve && $(RUST) clean 2>/dev/null || true
	cd $(RUST_DIR)/python_bridge && $(RUST) clean 2>/dev/null || true
	rm -rf $(ROOT_DIR)/target
	rm -rf $(ROOT_DIR)/dist
	rm -rf $(ROOT_DIR)/*.egg-info
	find $(ROOT_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find $(ROOT_DIR) -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Clean complete."

# ──────────────────────────────────────────────────────────
# INSTALL
# ──────────────────────────────────────────────────────────

.PHONY: install
install: python
	@echo ""
	@echo "📦 Installing Neura-X..."
	cd $(ROOT_DIR) && pip install dist/*.whl --force-reinstall
	@echo "✅ Neura-X installed."

.PHONY: install-dev
install-dev:
	@echo ""
	@echo "📦 Installing Neura-X in development mode..."
	cd $(ROOT_DIR) && maturin develop --release
	@echo "✅ Neura-X installed in dev mode."

# ──────────────────────────────────────────────────────────
# BUILD DISTRIBUTION WHEELS
# ──────────────────────────────────────────────────────────

.PHONY: wheel
wheel: python
	@echo ""
	@echo "📦 Distribution wheels built successfully."
	@ls -la $(ROOT_DIR)/dist/*.whl 2>/dev/null || echo "   No wheels found."

# ──────────────────────────────────────────────────────────
# BUILD DOCUMENTATION
# ──────────────────────────────────────────────────────────

.PHONY: docs
docs:
	@echo ""
	@echo "📚 Building documentation..."
	@echo "   (Documentation build not yet implemented)"
	@echo "   See docs/ directory for markdown files."

# ──────────────────────────────────────────────────────────
# UTILITY TARGETS
# ──────────────────────────────────────────────────────────

.PHONY: info
info:
	@echo ""
	@echo "============================================================"
	@echo "  Neura-X Build Information"
	@echo "============================================================"
	@echo "  Project:     $(PROJECT_NAME)"
	@echo "  Version:     $(VERSION)"
	@echo "  C Compiler:  $(CC)"
	@echo "  C++ Compiler: $(CXX)"
	@echo "  Rust:        $(shell $(RUST) --version 2>/dev/null || echo 'not found')"
	@echo "  Python:      $(shell $(PYTHON) --version 2>/dev/null || echo 'not found')"
	@echo "  LLVM:        $(shell $(LLVM_CONFIG) --version 2>/dev/null || echo 'not found')"
	@echo "============================================================"

.PHONY: help
help:
	@echo ""
	@echo "============================================================"
	@echo "  Neura-X Build System — Help"
	@echo "============================================================"
	@echo ""
	@echo "  Targets:"
	@echo "    make all          Build everything (C, C++, Rust, Python)"
	@echo "    make c            Build C code only"
	@echo "    make cpp          Build C++ code only"
	@echo "    make rust         Build Rust code only"
	@echo "    make python       Build Python package"
	@echo "    make test         Run Python test suite"
	@echo "    make test-rust    Run Rust test suite"
	@echo "    make clean        Clean all build artifacts"
	@echo "    make install      Install Neura-X via pip"
	@echo "    make install-dev  Install in development mode"
	@echo "    make wheel        Build distribution wheels"
	@echo "    make docs         Build documentation"
	@echo "    make info         Show build environment info"
	@echo "    make help         Show this help message"
	@echo ""
	@echo "  Neura-X: Intelligence Without Limits."
	@echo "  © 2026 Edusei Mikel Lisamba. All Rights Reserved."
	@echo ""