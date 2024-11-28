import os
import subprocess
import yaml


if __name__ == "__main__":
    print(os.getcwd())
    result = subprocess.run(["./start.sh"], text=True)
    print("=== stdout ===")
    print(result.stdout)
    print("=== stderr ===")
    print(result.stderr)
