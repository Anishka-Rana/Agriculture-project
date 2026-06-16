import os
import shutil

source_train = "dataset/train"
source_test = "dataset/test"

target_train = "final_dataset/train"
target_test = "final_dataset/test"

# create folders
for path in [target_train, target_test]:
    os.makedirs(os.path.join(path, "healthy"), exist_ok=True)
    os.makedirs(os.path.join(path, "diseased"), exist_ok=True)

def process_folder(source, target):
    for category in os.listdir(source):
        category_path = os.path.join(source, category)

        if not os.path.isdir(category_path):
            continue

        for folder in os.listdir(category_path):
            folder_path = os.path.join(category_path, folder)

            if not os.path.isdir(folder_path):
                continue

            # classify
            if "healthy" in folder.lower():
                target_folder = "healthy"
            else:
                target_folder = "diseased"

            count = 0
            for img in os.listdir(folder_path):
                if count >= 300:   # limit images (fast training)
                    break

                src = os.path.join(folder_path, img)
                dst = os.path.join(target, target_folder, img)

                if os.path.isfile(src):
                    try:
                        shutil.copy(src, dst)
                        count += 1
                    except:
                        pass

# run conversion
process_folder(source_train, target_train)
process_folder(source_test, target_test)

print("✅ Dataset converted successfully!")

