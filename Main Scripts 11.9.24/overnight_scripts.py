import subprocess

# List of scripts with their arguments
WORKING_PATH = "C:/Users/carlo/Projects/Astronomy Research/Main Scripts 11.9.24/"
scripts = [
    [f"{WORKING_PATH}6_emulator_training.py", "0.5", "1"],
    [f"{WORKING_PATH}6_emulator_training.py", "0.5", "10"],
    [f"{WORKING_PATH}6_emulator_training.py", "0.6", "1"],
    [f"{WORKING_PATH}6_emulator_training.py", "0.6", "10"],
    [f"{WORKING_PATH}6_emulator_training.py", "0.7", "1"],
    [f"{WORKING_PATH}6_emulator_training.py", "0.7", "10"],
    [f"{WORKING_PATH}6_emulator_training.py", "0.8", "1"],
    [f"{WORKING_PATH}6_emulator_training.py", "0.8", "10"],
    [f"{WORKING_PATH}6_emulator_training.py", "0.9", "1"],
    [f"{WORKING_PATH}6_emulator_training.py", "0.9", "10"],
    [f"{WORKING_PATH}6_emulator_training.py", "1", "1"],
    [f"{WORKING_PATH}6_emulator_training.py", "1", "10"]
]

# Run the scripts
for script in scripts: 
    print(f"Running {' '.join(script)}...")
    subprocess.run(["python"] + script, check=True)
    print(f"{' '.join(script)} completed.")