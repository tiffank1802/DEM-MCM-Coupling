"""Script to fix ANN annotations in test_session_sync.py."""

import re

filepath = "tests/test_session_sync.py"

with open(filepath) as f:
    content = f.read()
    lines = content.split("\n")

# Mapping of fixture parameter names to their types
fixture_types = {
    "app_context": "Any",
    "sample_loaded_model": "Any",
    "sample_transition_matrix": "Any",
    "mock_open": "MagicMock",
    "mock_fs": "MagicMock",
    "mock_streamlit": "MagicMock",
    "mock_session_state": "MagicMock",
    "mock_query_params": "MagicMock",
}

# Also fix fixture return types
fixture_return_types = {
    "sample_loaded_model": "Any",
    "sample_transition_matrix": "np.ndarray",
    "app_context": "Any",
    "mock_streamlit_session": "MagicMock",
}

fixes = 0

# Fix fixture functions that return None
for i, line in enumerate(lines):
    # Check for fixtures missing return type
    for name, ret_type in fixture_return_types.items():
        pattern = f"def {name}("
        if line.strip().startswith(pattern) and "->" not in line:
            # Check if this is a fixture function
            has_fixture_decorator = False
            for j in range(max(0, i - 3), i):
                if "@pytest.fixture" in lines[j] or "@fixture" in lines[j]:
                    has_fixture_decorator = True
                    break
            if has_fixture_decorator:
                lines[i] = line.rstrip() + f" -> {ret_type}:"
                fixes += 1

# Fix test function parameters that reference fixtures
# These have patterns like: def test_xxx(self, app_context, sample_loaded_model):
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("def ") and "self" in stripped:
        # Check if any fixture params are in this function's signature
        modified = False
        for param_name, param_type in fixture_types.items():
            # Match the parameter name followed by , or ) or =
            # But not already annotated
            old = rf"\b{param_name}\b(?!\s*:)"

            # Check if the param is present but not annotated
            if (
                re.search(rf"\b{param_name}\b", stripped)
                and param_name + ":" not in stripped
            ):
                # Replace param_name with param_name: param_type
                # But only in the function's parameter list
                new = rf"{param_name}: {param_type}"
                # Find the exact position in the line
                idx = stripped.find("def ")
                if idx >= 0:
                    params_start = stripped.find("(", idx)
                    params_end = stripped.rfind(")")
                    if params_start >= 0 and params_end > params_start:
                        params_section = stripped[params_start : params_end + 1]
                        # Check if param is in params section
                        if (
                            f" {param_name}," in params_section
                            or f" {param_name})" in params_section
                            or f" {param_name}=" in params_section
                        ):
                            lines[i] = lines[i].replace(
                                f" {param_name},", f" {param_name}: {param_type},"
                            )
                            lines[i] = lines[i].replace(
                                f" {param_name})", f" {param_name}: {param_type})"
                            )
                            lines[i] = lines[i].replace(
                                f" {param_name}=", f" {param_name}: {param_type}="
                            )
                            modified = True
                            fixes += 1
        if modified:
            pass

# Fix multi-line function signatures
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
lines = new_lines

content = "\n".join(lines)

# Add Any import if needed
if "from typing import " in content:
    if "Any" not in content.split("from typing import ")[1].split("\n")[0]:
        content = content.replace("from typing import ", "from typing import Any, ")
elif "import typing" not in content:
    # Add import at top
    lines = content.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("import ") or l.startswith("from "):
            if "typing" not in l:
                continue
            break
    else:
        # No typing import found, add one
        for i, l in enumerate(lines):
            if l.startswith("import ") or l.startswith("from "):
                if "from typing import" in content:
                    content = content.replace(
                        "from typing import ", "from typing import Any, "
                    )
                    break
                lines.insert(i, "from typing import Any")
                content = "\n".join(lines)
                break

# Add MagicMock import if needed (it's from unittest.mock)
if "from unittest.mock import " in content:
    if "MagicMock" not in content.split("from unittest.mock import ")[1].split("\n")[0]:
        content = content.replace(
            "from unittest.mock import ", "from unittest.mock import MagicMock, "
        )
elif "unittest.mock" not in content:
    lines = content.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("import ") or l.startswith("from "):
            continue
        if l.startswith("#") or l.startswith('"""') or l.startswith("'''"):
            continue
        if l.strip() == "":
            continue
        lines.insert(i, "from unittest.mock import MagicMock")
        content = "\n".join(lines)
        break

# Also add numpy import for np.ndarray
if "import numpy as np" not in content and "import numpy" not in content:
    lines = content.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("import ") or l.startswith("from "):
            continue
        if l.startswith("#") or l.startswith('"""') or l.startswith("'''"):
            continue
        if l.strip() == "":
            continue
        lines.insert(i, "import numpy as np")
        content = "\n".join(lines)
        break

with open(filepath, "w") as f:
    f.write(content)

print(f"Fixed {fixes} annotations.")
