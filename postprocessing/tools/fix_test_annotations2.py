"""Fix remaining type annotations in test_inhomogeneous_markov.py - round 2."""

import re

with open("tests/test_inhomogeneous_markov.py") as f:
    content = f.read()
    lines = content.split("\n")

# Additional fixture types
fixture_types = {
    "rng": "np.random.RandomState",
    "n_states": "int",
    "n_particles": "int",
    "n_timesteps": "int",
    "n_timesteps_large": "int",
    "n_blocks": "int",
    "homogeneous_config": "ExperimentConfig",
    "inhomogeneous_config": "ExperimentConfig",
    "inhomogeneous_config_single_nlt": "ExperimentConfig",
    "inhomogeneous_P_blocks": "np.ndarray",
    "homogeneous_transition_matrix": "np.ndarray",
    "synthetic_timestep_dict": "dict[int, pd.DataFrame]",
    "synthetic_timestep_dict_large": "dict[int, pd.DataFrame]",
    "synthetic_S_matrix": "np.ndarray",
    "synthetic_times": "np.ndarray",
    "mock_partitioner": "MockPartitioner",
    "mock_get_fs": "MagicMock",
    "mock_get_api": "MagicMock",
    "particle_diameter": "float | None",
}

# For each line, try to match and fix
new_lines = []
changes = 0
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    is_continuation = False

    # Check if this line starts a function definition
    m = re.match(r"^(def\s+\w+\s*\()(.*)$", stripped)
    if m:
        prefix = m.group(1)
        params_str = m.group(2)

        # Collect all parameter lines (handle multi-line defs)
        full_params = params_str
        while (
            not full_params.rstrip().endswith("):")
            and not full_params.rstrip().endswith(":")
            and not full_params.rstrip().endswith("-> None:")
        ):
            # Check if there's a return type annotation
            if "->" in full_params and full_params.rstrip().endswith(":"):
                break
            i += 1
            if i < len(lines):
                next_line = lines[i]
                full_params += " " + next_line.strip()

        # Now parse the full parameter string
        # Find the position of the closing parenthesis
        paren_depth = 0
        params_body = ""
        return_annotation = ""
        in_params = False
        for ch in full_params:
            if ch == "(":
                paren_depth += 1
                if paren_depth == 1:
                    in_params = True
                    continue
            if ch == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    in_params = False
                    continue
            if in_params:
                params_body += ch
            elif paren_depth == 0 and ch != ":":
                return_annotation += ch

        if params_body.strip() and not (
            "self)" in full_params and len(params_body.strip()) == 4
        ):
            # Parse individual params
            params = []
            current = ""
            depth = 0
            for ch in params_body:
                if ch in ("(", "["):
                    depth += 1
                    current += ch
                elif ch in (")", "]"):
                    depth -= 1
                    current += ch
                elif ch == "," and depth == 0:
                    params.append(current.strip())
                    current = ""
                else:
                    current += ch
            if current.strip():
                params.append(current.strip())

            new_params = []
            for p in params:
                if not p:
                    continue
                if p == "self":
                    new_params.append(p)
                    continue
                if ":" in p:
                    new_params.append(p)
                    continue
                if p.startswith("**") or p.startswith("*"):
                    new_params.append(p)
                    continue

                p_parts = p.split("=", 1)
                p_name = p_parts[0].strip()
                has_default = len(p_parts) > 1
                default_val = p_parts[1] if has_default else None

                ptype = fixture_types.get(p_name)
                if ptype:
                    if has_default:
                        if default_val.strip() == "None" and "| None" not in ptype:
                            if "| None" not in ptype:
                                ptype = f"{ptype} | None"
                        new_params.append(f"{p_name}: {ptype} = {default_val}")
                    else:
                        new_params.append(f"{p_name}: {ptype}")
                    changes += 1
                else:
                    new_params.append(p)

            params_body = ", ".join(new_params)

        # Rebuild the def line
        if return_annotation.strip():
            new_line = indent + "def " + full_params.split("def ", 1)[1]
        else:
            new_line = indent + "def " + full_params.split("def ", 1)[1]

        new_lines.append(new_line)
    else:
        new_lines.append(line)
    i += 1

result = "\n".join(new_lines)
with open("tests/test_inhomogeneous_markov.py", "w") as f:
    f.write(result)

print(f"Fixed {changes} additional parameter annotations")
