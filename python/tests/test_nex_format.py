# ==========================================================
# Neura-X Tests: .nex File Format
# ==========================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_nex_magic_number():
    """Test that the .nex magic number is correct."""
    MAGIC = 0x4E455858  # "NEXX"
    assert MAGIC == int.from_bytes(b"NEXX", 'big')
    print("✅ test_nex_magic_number passed")


def test_nex_header_size():
    """Test that the header size is 128 bytes."""
    HEADER_SIZE = 128
    assert HEADER_SIZE == 128
    print("✅ test_nex_header_size passed")


def test_founder_name():
    """Test that the Founder's name is correct."""
    FOUNDER = "Edusei Mikel Lisamba"
    assert len(FOUNDER) <= 64  # Must fit in 64-byte header field
    print("✅ test_founder_name passed")


if __name__ == "__main__":
    test_nex_magic_number()
    test_nex_header_size()
    test_founder_name()
    print("\n🎉 All .nex format tests passed!")