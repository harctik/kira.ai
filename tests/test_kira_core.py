"""
Unit tests for Kira OS Tools and Evolution Memory Engine.
"""

import os
import sys
import tempfile
import pytest

# Ensure Kira root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import LaptopTools
from evolution import EvolutionEngine


def test_laptop_tools_execute_shell_echo():
    output = LaptopTools.execute_shell("echo 'Kira online'")
    assert "Kira online" in output


def test_laptop_tools_read_file(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Hello from Kira\nLine 2\nLine 3", encoding="utf-8")

    content = LaptopTools.read_file(str(sample_file))
    assert "Hello from Kira" in content
    assert "Line 2" in content


def test_evolution_engine_memory_lifecycle(tmp_path):
    mem_dir = tmp_path / "test_memory"
    engine = EvolutionEngine(memory_dir=str(mem_dir))

    assert engine.profile["total_interactions"] == 0
    assert os.path.exists(str(mem_dir))
