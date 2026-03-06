"""
Experiment Memory -- ReMe file-based persistence for Yang-Mills campaigns.

Tracks explored beta ranges, lattice sizes, mass gap signals,
and experiment results across sessions.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional


class ExperimentMemory:
    """
    Persistent memory for Yang-Mills experiments.
    Uses MEMORY.md and daily logs (ReMe CoPaw pattern).
    """
    def __init__(self, working_dir: str = "."):
        self.working_dir = Path(working_dir)
        self.memory_file = self.working_dir / "MEMORY.md"
        self.memory_dir = self.working_dir / "memory"
        self.memory_dir.mkdir(exist_ok=True)

    def load_explored_betas(self) -> List[float]:
        """Load already-explored beta values from MEMORY.md."""
        betas = []
        if not self.memory_file.exists():
            return betas
        content = self.memory_file.read_text(encoding="utf-8")
        in_section = False
        for line in content.splitlines():
            if line.strip().startswith("## Explored Betas"):
                in_section = True
                continue
            if in_section and line.startswith("## "):
                break
            if in_section and line.startswith("- `beta="):
                try:
                    val = float(line.split("=")[1].split("`")[0])
                    betas.append(val)
                except (IndexError, ValueError):
                    continue
        return betas

    def save_experiment(self, beta: float, N: int, dim: int,
                         verdict: str, sigma: float,
                         plaquette: float, details: str = "") -> None:
        """Save an experiment result to daily log and update MEMORY.md."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        daily_log = self.memory_dir / f"{today}.md"

        entry = f"\n### Experiment: beta={beta}, N={N}, dim={dim}\n"
        entry += f"- **Timestamp**: {datetime.utcnow().isoformat()}Z\n"
        entry += f"- **Verdict**: {verdict}\n"
        entry += f"- **String tension (sigma)**: {sigma}\n"
        entry += f"- **Average plaquette**: {plaquette}\n"
        if details:
            entry += f"- **Details**: {details}\n"

        with open(daily_log, "a", encoding="utf-8") as f:
            if daily_log.stat().st_size == 0:
                f.write(f"# Yang-Mills Experiment Log -- {today}\n")
            f.write(entry)

        # Update MEMORY.md
        existing_betas = self.load_explored_betas()
        if beta not in existing_betas:
            existing_betas.append(beta)
        existing_betas.sort()

        self._update_memory(existing_betas, verdict, beta, sigma)

    def _update_memory(self, betas: List[float], latest_verdict: str,
                        latest_beta: float, latest_sigma: float) -> None:
        lines = []
        lines.append("# Yang-Mills Experiment Memory\n\n")
        lines.append(f"> Last updated: {datetime.utcnow().isoformat()}Z\n\n")

        lines.append("## Summary\n\n")
        lines.append(f"- **Total beta values explored**: {len(betas)}\n")
        lines.append(f"- **Beta range**: [{min(betas):.2f}, {max(betas):.2f}]\n")
        lines.append(f"- **Latest verdict**: {latest_verdict} at beta={latest_beta}\n")
        lines.append(f"- **Latest sigma**: {latest_sigma}\n\n")

        lines.append("## Explored Betas\n\n")
        for b in betas:
            lines.append(f"- `beta={b}`\n")

        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
