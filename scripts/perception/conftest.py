"""Pytest configuration for perception test directory.

Excludes GPU integration scripts that require Isaac Lab AppLauncher
and cannot run under plain pytest collection.
"""

collect_ignore = ["test_ekf_integration.py"]
