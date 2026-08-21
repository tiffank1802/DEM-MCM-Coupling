"""Fix missing type annotations in test_inhomogeneous_markov.py."""

import re

with open("tests/test_inhomogeneous_markov.py") as f:
    content = f.read()
    lines = content.split("\n")

# Parameter type mapping for fixtures (by name)
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
    "particle_diameter": "float | None",
    "particle_diameters": "np.ndarray | None",
    "use_diameter": "bool",
}

# For each function def line, find unannotated params and add types
new_lines = []
changes = 0

for i, line in enumerate(lines):
    stripped = line.lstrip()
    # Match function def lines
    m = re.match(r"^(def\s+\w+\s*\()(.*)(\)\s*->.*:|\s*\)\s*:)", stripped)
    if m:
        prefix = m.group(1)
        params_str = m.group(2)
        suffix = m.group(3)
        indent = line[: len(line) - len(stripped)]

        # If there's already "->" in the suffix, skip return type
        if "->" in suffix:
            pass  # Already has return type annotation

        # Parse params and add types
        if params_str.strip():
            params = [p.strip() for p in params_str.split(",")]
            new_params = []
            for p in params:
                # Skip 'self'
                if p == "self":
                    new_params.append(p)
                    continue
                # Check if param already has annotation
                if ":" in p:
                    new_params.append(p)
                    continue
                # Check if it's a **kwargs or *args
                if p.startswith("**") or p.startswith("*"):
                    continue
                # Extract param name and default value
                p_parts = p.split("=", 1)
                p_name = p_parts[0].strip()
                has_default = len(p_parts) > 1
                default_val = p_parts[1] if has_default else None

                # Find type for this parameter
                ptype = fixture_types.get(p_name)
                if ptype:
                    if has_default:
                        # Check if default is None -> make Optional
                        if default_val.strip() == "None":
                            if "| None" not in ptype:
                                ptype = f"{ptype} | None"
                        new_params.append(f"{p_name}: {ptype} = {default_val}")
                    else:
                        new_params.append(f"{p_name}: {ptype}")
                    changes += 1
                else:
                    new_params.append(p)
            params_str = ", ".join(new_params)

        new_line = indent + prefix + params_str + suffix
        new_lines.append(new_line)
    else:
        new_lines.append(line)

result = "\n".join(new_lines)
with open("tests/test_inhomogeneous_markov.py", "w") as f:
    f.write(result)

print(f"Fixed {changes} parameter annotations")
