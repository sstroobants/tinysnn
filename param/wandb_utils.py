import os

import wandb

def find_wandb_run_by_name(model_name, path):
    # get wandb api
    api = wandb.Api()

    # Get all runs in the project
    runs = api.runs(path)

    # Find the run by name
    target_run = None
    for run in runs:
        if run.name == model_name:
            target_run = run
            break
    if target_run is None:
        print(f"Run with name '{model_name}' not found.")
    else:
        print(f"Found run: {target_run.id} with name: {target_run.name}")
    return target_run

def download_model_for_run(target_run):
    # List all artifacts in the run
    found_model = False
    for art in target_run.used_artifacts():
        if "model" in art.name:
            print(art.name)
            found_model = True
            break
    if not found_model:
        for art in target_run.logged_artifacts():
            print(art.name)
            found_model = True
            break
    if found_model:
        # Define the download path and create it if it does not exist
        download_folder = f"param/models/{target_run.name}"
        os.makedirs(download_folder, exist_ok=True)
        artifact_dir = art.download(root=download_folder)
        print(artifact_dir)
    return target_run