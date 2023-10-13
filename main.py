import subprocess
import os

def run_script(script_name):
    """
    Executes the given Python script using subprocess.
    """
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {script_name}!")
        print(result.stderr)
    else:
        print(f"{script_name} executed successfully.")
    return result.returncode

def main():
    # Set the path to where your scripts are located
    script_directory = '/Users/heejin/Library/CloudStorage/OneDrive-SNU/Documents/Topics/Computational Analysis/Programs/MyPrograms/CharacterNetwork'
    os.chdir(script_directory)  # Change the working directory

    scripts_to_run = [
        'CharacterNetworkBulk.py',
        'NetworkVisualizationBulk.py',
        'CharacterCentralityMetrics.py',
        'CharacterCentralityBarGraphs.py'
    ]

    for script in scripts_to_run:
        return_code = run_script(script)
        if return_code != 0:
            # If you want to stop execution on error:
            break

if __name__ == "__main__":
    main()
