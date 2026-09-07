// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// LLVM JIT Engine — Header
// ==========================================================

#ifndef NEURA_X_LLVM_JIT_H
#define NEURA_X_LLVM_JIT_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_map>

namespace neurax {
namespace jit {

// ──────────────────────────────────────────────────────────
// Fractal Seed Parameters
// ──────────────────────────────────────────────────────────

struct FractalSeedParams {
    std::vector<float> a;     // Amplitudes
    std::vector<float> w1;    // Row frequencies
    std::vector<float> w2;    // Column frequencies
    std::vector<float> phi;   // Phase shifts
    int K;                    // Number of harmonics

    FractalSeedParams() : K(0) {}
    FractalSeedParams(int k) : a(k), w1(k), w2(k), phi(k), K(k) {}
};

// ──────────────────────────────────────────────────────────
// Compiled Function Handle
// ──────────────────────────────────────────────────────────

using BloomFunction = std::function<void(
    int row_start, int row_end, int cols,
    float* output
)>;

struct CompiledKernel {
    std::string name;
    BloomFunction function;
    size_t code_size_bytes;
    bool valid;

    CompiledKernel() : code_size_bytes(0), valid(false) {}
};

// ──────────────────────────────────────────────────────────
// JIT Engine
// ──────────────────────────────────────────────────────────

class JITEngine {
public:
    JITEngine();
    ~JITEngine();

    // Initialize the JIT engine. Returns 0 on success.
    int initialize();

    // Compile a Fractal Seed into a native bloom function.
    // Returns a CompiledKernel that can generate weights on-the-fly.
    CompiledKernel compile_fractal_bloom(
        const std::string& name,
        const FractalSeedParams& seed
    );

    // Compile a matrix multiplication kernel for a specific size.
    CompiledKernel compile_matmul(
        const std::string& name,
        int rows, int cols, int depth
    );

    // Get a cached kernel by name.
    CompiledKernel* get_cached_kernel(const std::string& name);

    // Clear all cached kernels.
    void clear_cache();

    // Get the number of cached kernels.
    size_t cache_size() const;

    // Check if the engine is initialized.
    bool is_initialized() const { return initialized_; }

    // Get the target triple (e.g., "x86_64-unknown-linux-gnu").
    std::string target_triple() const;

    // Get the CPU features string.
    std::string cpu_features() const;

private:
    bool initialized_;
    std::string target_triple_;
    std::string cpu_features_;
    std::unordered_map<std::string, CompiledKernel> kernel_cache_;

    // Internal: Generate the bloom function body.
    BloomFunction generate_bloom_impl(const FractalSeedParams& seed);
};

// ──────────────────────────────────────────────────────────
// Fractal Compiler (Higher-Level Interface)
// ──────────────────────────────────────────────────────────

class FractalCompiler {
public:
    FractalCompiler();
    ~FractalCompiler();

    // Initialize with a JIT engine.
    int initialize(JITEngine* engine);

    // Compile a Fractal Tensor for on-the-fly weight generation.
    CompiledKernel compile(
        const std::string& name,
        int rows, int cols,
        const FractalSeedParams& seed
    );

    // Get compilation statistics.
    struct CompileStats {
        int total_compilations;
        int cache_hits;
        int cache_misses;
        double total_compile_time_ms;
    };

    CompileStats get_stats() const;

private:
    JITEngine* engine_;
    CompileStats stats_;
};

} // namespace jit
} // namespace neurax

#endif // NEURA_X_LLVM_JIT_H