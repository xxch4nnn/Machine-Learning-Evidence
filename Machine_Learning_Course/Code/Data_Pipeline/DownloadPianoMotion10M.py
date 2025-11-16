import os
import git

def download_pianomotion10m(repo_url, download_dir):
    """
    Clones the PianoMotion10M dataset from its GitHub repository.

    Args:
        repo_url (str): The URL of the GitHub repository.
        download_dir (str): The local directory to clone the repository into.
    """
    if not os.path.exists(download_dir):
        print(f"Creating directory: {download_dir}")
        os.makedirs(download_dir)

    if not os.listdir(download_dir):
        print(f"Cloning repository from {repo_url} into {download_dir}...")
        try:
            git.Repo.clone_from(repo_url, download_dir)
            print("Download complete.")
        except git.exc.GitCommandError as e:
            print(f"Error cloning repository: {e}")
    else:
        print("Dataset already exists in the specified directory.")

if __name__ == "__main__":
    REPO_URL = "https://github.com/agnJason/PianoMotion10M"
    DOWNLOAD_DIR = "Machine_Learning_Course/Data/PianoMotion10M"
    download_pianomotion10m(REPO_URL, DOWNLOAD_DIR)
