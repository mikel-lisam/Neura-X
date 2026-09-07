// ==========================================================
// Neura-X: Intelligence Without Limits.
// Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
//
// Fractal Compiler — High-level compilation interface
// ==========================================================

#include "llvm_jit.h"
#include <chrono>
#include <sstream>

namespace neurax {
namespace jit {

FractalCompiler::FractalCompiler()
    : engine_(nullptr)
{
    stats_.total_compilations = 0;
    stats_.cache_hits = 0;
    stats_.cache_misses = 0;
    stats_.total_compile_time_ms = 0.0;
}

FractalCompiler::~FractalCompiler() {}

int FractalCompiler::initialize(JITEngine* engine)
{
    if (!engine) return -1;
    engine_ = engine;
    return 0;
}

CompiledKernel FractalCompiler::compile(
    const std::string& name,
    int rows, int cols,
    const FractalSeedParams& seed)
{
    // Suppress unused parameter warnings (dimensions handled by JIT engine)
    (void)rows;
    (void)cols;

    if (!engine_) {
        CompiledKernel invalid;
        invalid.valid = false;
        return invalid;
    }

    auto start = std::chrono::high_resolution_clock::now();

    CompiledKernel* cached = engine_->get_cached_kernel(name);
    if (cached) {
        stats_.cache_hits++;
        return *cached;
    }

    stats_.cache_misses++;
    CompiledKernel kernel = engine_->compile_fractal_bloom(name, seed);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    stats_.total_compilations++;
    stats_.total_compile_time_ms += elapsed_ms;

    return kernel;
}

FractalCompiler::CompileStats FractalCompiler::get_stats() const
{
    return stats_;
}

} // namespace jit
} // namespace neurax