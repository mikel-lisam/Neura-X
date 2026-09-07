#!/usr/bin/env python3
"""
Neura-X Smoke Test
Validates that the framework imports and core bindings work.
"""

import sys

def test_import():
    """Test that neura_x imports successfully."""
    try:
        import neura_x as nx
        print("✅ PASS: neura_x imported successfully")
        print(f"   Version: {nx.__version__}")
        print(f"   Author:  {nx.__author__}")
        print(f"   Tagline: {nx.__tagline__}")
        return True
    except Exception as e:
        print(f"❌ FAIL: Import failed: {e}")
        return False

def test_rust_core():
    """Test that the Rust core bindings are accessible."""
    try:
        from neura_x import _core
        version = _core.get_version()
        founder = _core.get_founder()
        tagline = _core.get_tagline()
        print(f"✅ PASS: Rust core bindings loaded")
        print(f"   Core Version: {version}")
        print(f"   Core Founder: {founder}")
        print(f"   Core Tagline: {tagline}")
        return True
    except Exception as e:
        print(f"❌ FAIL: Rust core bindings failed: {e}")
        return False

def test_founders_lock():
    """Test the Founder's Lock cryptographic signature."""
    try:
        from neura_x import _core
        signature = _core.get_founders_lock_signature()
        assert len(signature) == 64, "SHA-256 hex should be 64 chars"
        assert signature.isalnum(), "Signature should be alphanumeric hex"
        print(f"✅ PASS: Founder's Lock generated")
        print(f"   Signature: {signature[:32]}...")
        return True
    except Exception as e:
        print(f"❌ FAIL: Founder's Lock failed: {e}")
        return False

def test_cpu_info():
    """Test CPU detection from the C HAL."""
    try:
        from neura_x import _core
        info = _core.get_cpu_info()
        assert "Edusei Mikel Lisamba" in info
        print(f"✅ PASS: CPU info retrieved")
        return True
    except Exception as e:
        print(f"❌ FAIL: CPU info failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  ⚡ Neura-X Smoke Test")
    print("     Intelligence Without Limits.")
    print("=" * 60)
    print()

    results = []
    results.append(test_import())
    print()
    results.append(test_rust_core())
    print()
    results.append(test_founders_lock())
    print()
    results.append(test_cpu_info())
    print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"  🎉 ALL {total} SMOKE TESTS PASSED!")
    else:
        print(f"  ⚠️  {passed}/{total} tests passed. {total - passed} failed.")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)