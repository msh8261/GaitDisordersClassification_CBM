import os
import sys

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import glob

import config.config_data as config

dir_dist = config.params["dir_dist"]


def merge_and_save_files(files, dst_data_path):
    # Reading data from file1
    with open(dst_data_path, "w") as fw:
        for file in files:
            with open(file, "r") as fr:
                data = fr.read()
                fw.write(data)


def merge_and_save_train_labels(files, dst_label_path):
    # Reading data from file1
    with open(dst_label_path, "w") as fw:
        for file in files:
            with open(file, "r") as fr:
                data = fr.read()
                fw.write(data)


def merge_and_save_test_labels(files, dst_label_path):
    # Reading data from file1
    with open(dst_label_path, "w") as fw:
        for file in files:
            with open(file, "r") as fr:
                data = fr.read()
                fw.write(data)
                fw.write("\n")


def main():

    base_folders = [
        name
        for name in os.listdir(dir_dist)
        if os.path.isdir(os.path.join(dir_dist, name))
    ]

    list_data_split = []
    list_label_split = []
    list_data_val = []
    list_label_val = []
    for folder_patient_id in base_folders:
        base_dir1 = os.path.join(dir_dist, folder_patient_id)
        folders_patient_date = [
            name
            for name in os.listdir(base_dir1)
            if os.path.isdir(os.path.join(base_dir1, name))
        ]
        # data_Files_path = [os.path.join(base_dir1, name) for name in os.listdir(base_dir1) if name.endswith('.File')]
        for folder_date in folders_patient_date:
            data_val_path = (
                base_dir1
                + "/"
                + folder_patient_id
                + "_"
                + folder_date
                + "_data_val.File"
            )
            data_label_path = (
                base_dir1 + "/" + folder_patient_id + "_" + folder_date + "_label.File"
            )
            if os.path.isfile(data_val_path):
                if os.path.isfile(data_label_path):
                    list_data_val.append(data_val_path)
                    list_label_val.append(data_label_path)
                    continue

            data_split_path = (
                base_dir1
                + "/"
                + folder_patient_id
                + "_"
                + folder_date
                + "_data_train_split.File"
            )
            label_split_path = (
                base_dir1
                + "/"
                + folder_patient_id
                + "_"
                + folder_date
                + "_label_split.File"
            )

            if os.path.isfile(data_split_path):
                if os.path.isfile(label_split_path):
                    list_data_split.append(data_split_path)
                    list_label_split.append(label_split_path)

    dst_data_train_path = os.path.join(dir_dist, "Xtrain.File")
    dst_label_train_path = os.path.join(dir_dist, "ytrain.File")
    merge_and_save_files(list_data_split, dst_data_train_path)
    merge_and_save_train_labels(list_label_split, dst_label_train_path)

    dst_data_val_path = os.path.join(dir_dist, "Xtest.File")
    dst_label_val_path = os.path.join(dir_dist, "ytest.File")
    merge_and_save_files(list_data_val, dst_data_val_path)
    merge_and_save_test_labels(list_label_val, dst_label_val_path)


if __name__ == "__main__":
    main()
