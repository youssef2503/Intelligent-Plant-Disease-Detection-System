import os
import shutil

dataset_dir = 'dataset'

def remove_potato_classes():
    if not os.path.exists(dataset_dir):
        print("Dataset folder not found!")
        return

    folders = os.listdir(dataset_dir)
    deleted_count = 0

    for folder in folders:
        if 'Potato' in folder:
            folder_path = os.path.join(dataset_dir, folder)
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted: {folder}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {folder}: {e}")

    print(f"\nDone! Deleted {deleted_count} Potato folders.")

if __name__ == "__main__":
    remove_potato_classes()