"""Run the current cumulative-rewards table and save it as an immutable reference."""
import subprocess
import sys

OUT = "docs/superpowers/baselines/2026-06-30-pre-fix-table.txt"

if __name__ == "__main__":
    res = subprocess.run(
        [sys.executable, "extract_cumulative_rewards_table.py"],
        capture_output=True, text=True,
    )
    with open(OUT, "w") as f:
        f.write(res.stdout)
        if res.stderr:
            f.write("\n# STDERR\n" + res.stderr)
    print(f"Wrote {OUT}")
    print(res.stdout)
