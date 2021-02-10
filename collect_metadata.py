import os

with open("vlsp2020_train_set_02/metadata.txt", "w") as new_file:
    for file in os.listdir("vlsp2020_train_set_02/"):
        if ".txt" in file:
            with open(file, "r") as old_file:
                text = old_file.readlines()[0]
                new_file.write(file.replace(".txt",".wav") + "|" + text)