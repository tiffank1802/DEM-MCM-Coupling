"""Fix ANN annotations in test_session_sync.py."""

filepath = "tests/test_session_sync.py"

with open(filepath) as f:
    content = f.read()
    lines = content.split("\n")

# Mapping of fixture parameter names to their concrete types (already imported)
fixture_types = {
    "app_context": "AppContext",
    "sample_loaded_model": "LoadedModel",
    "sample_transition_matrix": "np.ndarray",
    "sample_transition_model": "np.ndarray",
    "mock_open": "MagicMock",
    "mock_fs": "MagicMock",
    "mock_streamlit": "MagicMock",
    "mock_session_state": "MagicMock",
    "mock_query_params": "MagicMock",
}

# Fixture functions that need return types
fixture_return_types = {
    "sample_loaded_model": "LoadedModel",
    "sample_transition_matrix": "np.ndarray",
    "app_context": "AppContext",
    "sample_loaded_model": "LoadedModel",
    "mock_streamlit_session": "MagicMock",
}

fixes = 0

# Track if we need to add imports
needs_any = False
needs_magicmock = False

# Fix fixture functions missing return types
for i, line in enumerate(lines):
    stripped = line.strip()
    for name, ret_type in fixture_return_types.items():
        if stripped.startswith(f"def {name}(") and "->" not in stripped:
            # Check if preceded by @pytest.fixture or @fixture
            for j in range(max(0, i - 3), i):
                if "@pytest.fixture" in lines[j] or "@fixture" in lines[j]:
                    lines[i] = line.rstrip() + f" -> {ret_type}:"
                    fixes += 1
                    break

# Fix function parameters (handle both single-line and multi-line defs)
# Strategy: find all def lines, then examine parameters for unannotated fixtures
i = 0
while i < len(lines):
    stripped = lines[i].strip()
    if stripped.startswith("def ") and "self" in stripped and "->" in stripped:
        # Already has return type - might still need param annotations
        # Check for unannotated fixture params
        for param_name in fixture_types:
            if param_name in stripped and f"{param_name}:" not in stripped:
                # Need to add annotation
                lines[i] = lines[i].replace(
                    f" {param_name},", f" {param_name}: {fixture_types[param_name]},"
                )
                lines[i] = lines[i].replace(
                    f" {param_name})", f" {param_name}: {fixture_types[param_name]})"
                )
                lines[i] = lines[i].replace(
                    f" {param_name}=", f" {param_name}: {fixture_types[param_name]}="
                )
                fixes += 1
    elif stripped.startswith("def ") and "self" in stripped and "->" not in stripped:
        # Check for params that need annotation or return type
        for param_name in fixture_types:
            if param_name in stripped and f"{param_name}:" not in stripped:
                lines[i] = lines[i].replace(
                    f" {param_name},", f" {param_name}: {fixture_types[param_name]},"
                )
                lines[i] = lines[i].replace(
                    f" {param_name})", f" {param_name}: {fixture_types[param_name]})"
                )
                lines[i] = lines[i].replace(
                    f" {param_name}=", f" {param_name}: {fixture_types[param_name]}="
                )
                fixes += 1
    i += 1

content = "\n".join(lines)

# Ensure imports exist
if "from typing import Any" not in content:
    if "from typing import " in content:
        content = content.replace("from typing import ", "from typing import Any, ")
    else:
        # Add after last import
        lines = content.split("\n")
        last_import = -1
        for i, l in enumerate(lines):
            if l.startswith("import ") or l.startswith("from "):
                last_import = i
        if last_import >= 0:
            lines.insert(last_import + 1, "from typing import Any")
        content = "\n".join(lines)

if (
    "from unittest.mock import MagicMock" not in content
    and "from unittest.mock import " in content
):
    content = content.replace(
        "from unittest.mock import ", "from unittest.mock import MagicMock, "
    )
elif "MagicMock" in content and "from unittest.mock import MagicMock" not in content:
    # Need to add the import
    lines = content.split("\n")
    last_import = -1
    for i, l in enumerate(lines):
        if l.startswith("import ") or l.startswith("from "):
            last_import = i
    if last_import >= 0:
        if "from unittest.mock import patch" in content:
            # Already has patch import, add MagicMock
            for i, l in enumerate(lines):
                if l.strip().startswith("from unittest.mock import patch"):
                    lines[i] = l.replace(
                        "from unittest.mock import patch",
                        "from unittest.mock import MagicMock, patch",
                    )
                    break
        else:
            lines.insert(last_import + 1, "from unittest.mock import MagicMock")
    content = "\n".join(lines)

with open(filepath, "w") as f:
    f.write(content)

print(f"Fixed {fixes} annotations.")
