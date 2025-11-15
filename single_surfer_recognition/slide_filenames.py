import os

def slide(folder_path):
    """
    Retrieves a list of filenames (excluding directories) from a specified folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list: A list of strings, where each string is a filename within the folder.
    """
    filenames = []
    try:
        # Get all entries (files and directories) in the folder
        all_entries = os.listdir(folder_path)

        # Filter out only the files
        for entry in all_entries:
            full_path = os.path.join(folder_path, entry)
            if os.path.isfile(full_path):
                filenames.append(entry)
    except FileNotFoundError:
        print(f"Error: Folder not found at '{folder_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

    for i in range(len(filenames)):
        if (int(filenames[i][0:5]) != i):
            os.rename(f"{folder_path}/{filenames[i]}", f"{folder_path}/{i:05d}.jpg")

    print("finished closing filename gaps.")

if __name__ == "__main__":
    slide("single_surfer_recognition/training_photos")