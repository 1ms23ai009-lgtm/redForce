"""Exploit isolation environment for safe attack execution."""

import tempfile
import os
from contextlib import contextmanager


@contextmanager
def sandboxed_execution():
    """Context manager for sandboxed exploit testing.

    Creates an isolated temporary directory for any file operations
    and cleans up afterward.
    """
    sandbox_dir = tempfile.mkdtemp(prefix="redforge_sandbox_")
    original_dir = os.getcwd()
    try:
        os.chdir(sandbox_dir)
        yield sandbox_dir
    finally:
        os.chdir(original_dir)
        # Cleanup
        for root, dirs, files in os.walk(sandbox_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(sandbox_dir)
