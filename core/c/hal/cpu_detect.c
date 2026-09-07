/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * CPU Detection — Identifies CPU vendor, features, and cache
 * ========================================================== */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hal.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define NEX_ARCH_X86 1
    #if defined(_MSC_VER)
        #include <intrin.h>
    #elif defined(__GNUC__) || defined(__clang__)
        #include <cpuid.h>
    #endif
#elif defined(__aarch64__) || defined(__arm__)
    #define NEX_ARCH_ARM 1
    #include <sys/utsname.h>
#endif

#if defined(__linux__)
    #include <unistd.h>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
    #include <sys/types.h>
#elif defined(_WIN32)
    #include <windows.h>
#endif

/* ──────────────────────────────────────────────────────────
 * Internal: x86 CPUID wrapper
 * ────────────────────────────────────────────────────────── */

#ifdef NEX_ARCH_X86

static void nex_cpuid(int leaf, int subleaf, unsigned int *eax,
                      unsigned int *ebx, unsigned int *ecx, unsigned int *edx)
{
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, leaf, subleaf);
    *eax = (unsigned int)regs[0];
    *ebx = (unsigned int)regs[1];
    *ecx = (unsigned int)regs[2];
    *edx = (unsigned int)regs[3];
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#else
    *eax = *ebx = *ecx = *edx = 0;
#endif
}

static void nex_detect_x86(nex_cpu_info_t *info)
{
    unsigned int eax, ebx, ecx, edx;

    /* Get vendor string */
    nex_cpuid(0, 0, &eax, &ebx, &ecx, &edx);

    char vendor[13];
    memset(vendor, 0, sizeof(vendor));
    memcpy(vendor + 0, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);

    strncpy(info->vendor_string, vendor, 15);
    info->vendor_string[15] = '\0';

    if (strncmp(vendor, "GenuineIntel", 12) == 0) {
        info->vendor = NEX_CPU_VENDOR_INTEL;
    } else if (strncmp(vendor, "AuthenticAMD", 12) == 0) {
        info->vendor = NEX_CPU_VENDOR_AMD;
    } else {
        info->vendor = NEX_CPU_VENDOR_UNKNOWN;
    }

    /* Get model name (CPUID leaf 0x80000002 - 0x80000004) */
    char model[49];
    memset(model, 0, sizeof(model));
    for (unsigned int i = 0; i < 3; i++) {
        nex_cpuid(0x80000002 + i, 0, &eax, &ebx, &ecx, &edx);
        memcpy(model + i * 16 + 0,  &eax, 4);
        memcpy(model + i * 16 + 4,  &ebx, 4);
        memcpy(model + i * 16 + 8,  &ecx, 4);
        memcpy(model + i * 16 + 12, &edx, 4);
    }
    strncpy(info->model_name, model, 63);
    info->model_name[63] = '\0';

    /* Detect generation from model name (simplified) */
    info->generation = 0;
    if (info->vendor == NEX_CPU_VENDOR_INTEL) {
        /* Extract generation from model name like "i7-5600U" → 5 */
        char *dash = strchr(info->model_name, '-');
        if (dash && dash[1] >= '0' && dash[1] <= '9') {
            info->generation = dash[1] - '0';
        }
    }

    /* Get feature flags from CPUID leaf 1 */
    nex_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    info->features.sse2   = (edx >> 26) & 1;
    info->features.sse4_1 = (ecx >> 19) & 1;
    info->features.sse4_2 = (ecx >> 20) & 1;
    info->features.avx    = (ecx >> 28) & 1;
    info->features.fma    = (ecx >> 12) & 1;

    /* Get extended features from CPUID leaf 7 */
    nex_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    info->features.avx2     = (ebx >> 5)  & 1;
    info->features.bmi1     = (ebx >> 3)  & 1;
    info->features.bmi2     = (ebx >> 8)  & 1;
    info->features.avx512f  = (ebx >> 16) & 1;
    info->features.avx512bw = (ebx >> 30) & 1;
    info->features.avx512vl = (ebx >> 31) & 1;

    /* Determine SIMD width */
    if (info->features.avx512f) {
        info->simd_width_bytes = 64;
    } else if (info->features.avx2 || info->features.avx) {
        info->simd_width_bytes = 32;
    } else if (info->features.sse2) {
        info->simd_width_bytes = 16;
    } else {
        info->simd_width_bytes = 8;
    }

    /* Core count */
#if defined(__linux__)
    info->logical_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    info->physical_cores = info->logical_cores; /* Simplified */
#elif defined(__APPLE__)
    int count = 0;
    size_t count_len = sizeof(count);
    sysctlbyname("hw.physicalcpu", &count, &count_len, NULL, 0);
    info->physical_cores = count;
    sysctlbyname("hw.logicalcpu", &count, &count_len, NULL, 0);
    info->logical_cores = count;
#elif defined(_WIN32)
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    info->logical_cores = (int)sys_info.dwNumberOfProcessors;
    info->physical_cores = info->logical_cores;
#endif

    /* Cache sizes (CPUID leaf 4 for Intel, leaf 0x8000001D for AMD) */
    if (info->vendor == NEX_CPU_VENDOR_INTEL) {
        for (int i = 0; i < 4; i++) {
            nex_cpuid(4, i, &eax, &ebx, &ecx, &edx);
            int type = eax & 0x1F;
            if (type == 0) break;
            int level = (eax >> 5) & 0x7;
            uint64_t size = (uint64_t)((ebx >> 22) + 1) *
                            (((ebx >> 12) & 0x3FF) + 1) *
                            ((ebx & 0xFFF) + 1) * (ecx + 1);
            if (level == 1 && type == 1) info->l1_cache_bytes = size;
            if (level == 2) info->l2_cache_bytes = size;
            if (level == 3) info->l3_cache_bytes = size;
        }
    }
}

#endif /* NEX_ARCH_X86 */

/* ──────────────────────────────────────────────────────────
 * Internal: ARM detection
 * ────────────────────────────────────────────────────────── */

#ifdef NEX_ARCH_ARM

static void nex_detect_arm(nex_cpu_info_t *info)
{
    info->vendor = NEX_CPU_VENDOR_ARM;
    strncpy(info->vendor_string, "ARM", 15);

    struct utsname uname_data;
    if (uname(&uname_data) == 0) {
        strncpy(info->model_name, uname_data.machine, 63);
        info->model_name[63] = '\0';
    }

    /* ARM always has NEON on aarch64 */
    info->features.neon = true;
    info->features.avx  = false;
    info->features.avx2 = false;
    info->simd_width_bytes = 16; /* NEON is 128-bit */

    /* Detect Apple Silicon */
#if defined(__APPLE__)
    char hw_model[64] = {0};
    size_t len = sizeof(hw_model);
    if (sysctlbyname("hw.model", hw_model, &len, NULL, 0) == 0) {
        if (strstr(hw_model, "Mac") != NULL) {
            info->vendor = NEX_CPU_VENDOR_APPLE;
            strncpy(info->vendor_string, "Apple", 15);
            info->features.amx = true;
        }
    }

    int count = 0;
    size_t count_len = sizeof(count);
    sysctlbyname("hw.physicalcpu", &count, &count_len, NULL, 0);
    info->physical_cores = count;
    sysctlbyname("hw.logicalcpu", &count, &count_len, NULL, 0);
    info->logical_cores = count;

    /* Cache sizes on macOS */
    uint64_t cache_size = 0;
    len = sizeof(cache_size);
    sysctlbyname("hw.l1dcachesize", &cache_size, &len, NULL, 0);
    info->l1_cache_bytes = cache_size;
    sysctlbyname("hw.l2cachesize", &cache_size, &len, NULL, 0);
    info->l2_cache_bytes = cache_size;
    sysctlbyname("hw.l3cachesize", &cache_size, &len, NULL, 0);
    info->l3_cache_bytes = cache_size;
#else
    info->logical_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    info->physical_cores = info->logical_cores;
#endif
}

#endif /* NEX_ARCH_ARM */

/* ──────────────────────────────────────────────────────────
 * Public API
 * ────────────────────────────────────────────────────────── */

int nex_cpu_detect(nex_cpu_info_t *info)
{
    if (!info) return -1;

    memset(info, 0, sizeof(nex_cpu_info_t));

#ifdef NEX_ARCH_X86
    nex_detect_x86(info);
#elif defined(NEX_ARCH_ARM)
    nex_detect_arm(info);
#else
    info->vendor = NEX_CPU_VENDOR_UNKNOWN;
    strncpy(info->vendor_string, "Unknown", 15);
    info->logical_cores = 1;
    info->physical_cores = 1;
    info->simd_width_bytes = 8;
#endif

    return 0;
}

void nex_cpu_print_info(const nex_cpu_info_t *info)
{
    if (!info) return;

    printf("============================================================\n");
    printf("  Neura-X CPU Detection\n");
    printf("  Intelligence Without Limits.\n");
    printf("============================================================\n");
    printf("  Vendor:          %s\n", info->vendor_string);
    printf("  Model:           %s\n", info->model_name);
    printf("  Generation:      %d\n", info->generation);
    printf("  Physical Cores:  %d\n", info->physical_cores);
    printf("  Logical Cores:   %d\n", info->logical_cores);
    printf("  L1 Cache:        %llu KB\n",
           (unsigned long long)(info->l1_cache_bytes / 1024));
    printf("  L2 Cache:        %llu KB\n",
           (unsigned long long)(info->l2_cache_bytes / 1024));
    printf("  L3 Cache:        %llu KB\n",
           (unsigned long long)(info->l3_cache_bytes / 1024));
    printf("  SIMD Width:      %d bytes\n", info->simd_width_bytes);
    printf("────────────────────────────────────────────────────────────\n");
    printf("  Features:\n");
    printf("    SSE2:     %s\n", info->features.sse2     ? "✓" : "✗");
    printf("    SSE4.1:   %s\n", info->features.sse4_1   ? "✓" : "✗");
    printf("    SSE4.2:   %s\n", info->features.sse4_2   ? "✓" : "✗");
    printf("    AVX:      %s\n", info->features.avx      ? "✓" : "✗");
    printf("    AVX2:     %s\n", info->features.avx2     ? "✓" : "✗");
    printf("    AVX-512F: %s\n", info->features.avx512f  ? "✓" : "✗");
    printf("    FMA:      %s\n", info->features.fma      ? "✓" : "✗");
    printf("    NEON:     %s\n", info->features.neon     ? "✓" : "✗");
    printf("    AMX:      %s\n", info->features.amx      ? "✓" : "✗");
    printf("    BMI1:     %s\n", info->features.bmi1     ? "✓" : "✗");
    printf("    BMI2:     %s\n", info->features.bmi2     ? "✓" : "✗");
    printf("============================================================\n");
}

int nex_cpu_get_simd_width(const nex_cpu_info_t *info)
{
    if (!info) return 8;
    return info->simd_width_bytes;
}

bool nex_cpu_has_feature(const nex_cpu_info_t *info, const char *feature_name)
{
    if (!info || !feature_name) return false;

    if (strcmp(feature_name, "sse2") == 0)     return info->features.sse2;
    if (strcmp(feature_name, "sse4_1") == 0)   return info->features.sse4_1;
    if (strcmp(feature_name, "sse4_2") == 0)   return info->features.sse4_2;
    if (strcmp(feature_name, "avx") == 0)      return info->features.avx;
    if (strcmp(feature_name, "avx2") == 0)     return info->features.avx2;
    if (strcmp(feature_name, "avx512f") == 0)  return info->features.avx512f;
    if (strcmp(feature_name, "avx512bw") == 0) return info->features.avx512bw;
    if (strcmp(feature_name, "avx512vl") == 0) return info->features.avx512vl;
    if (strcmp(feature_name, "fma") == 0)      return info->features.fma;
    if (strcmp(feature_name, "neon") == 0)     return info->features.neon;
    if (strcmp(feature_name, "amx") == 0)      return info->features.amx;
    if (strcmp(feature_name, "bmi1") == 0)     return info->features.bmi1;
    if (strcmp(feature_name, "bmi2") == 0)     return info->features.bmi2;

    return false;
}

const char *nex_hal_version(void)
{
    return NEX_HAL_VERSION_STRING;
}

const char *nex_hal_founder(void)
{
    return NEX_FOUNDER;
}