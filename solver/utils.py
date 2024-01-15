import os
import shutil

def create_or_clear_folder(folder_path):
    """
    Deletes all files and subdirectories in the specified folder path if it exists, 
    excluding .md files. If the folder does not exist, creates a new folder at the specified path.

    Parameters:
        folder_path (str): The path of the folder to delete or create.

    Returns:
        None
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if the file is a regular file or a symbolic link
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    # Skip deletion of .md files
                    if not filename.endswith('.md'):
                        os.unlink(file_path)
                # Check if the file is a directory
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder_path)