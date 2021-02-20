import os
import soundfile as sf

'''
with open("../vlsp2020_train_set_02/metadata.txt", "w") as new_file:
    for file in os.listdir("../vlsp2020_train_set_02/"):
        if ".txt" in file and file != "metadata.txt":
            with open("../vlsp2020_train_set_02/" + file, "r") as old_file:
                f = sf.SoundFile("../vlsp2020_train_set_02/" + file)
                print(len(f) / f.samplerate)
                if len(f) / f.samplerate < 1.0:
                    continue
                text = old_file.readlines()[0]
                new_file.write(file.replace(".txt",".wav") + "|" + text + "\n")'''

with open("../metadata-4.txt", "r") as old_file:
  with open("../new_metadata.txt", "w") as new_file:
    for line in old_file.readlines():
      wav,text = line.split("|")
      f = sf.SoundFile("../vlsp2020_train_set_02/" + wav)
      #print(len(f) / f.samplerate)
      mi,mx = 0.05,0.08
      if len(f) / f.samplerate < 1.0:
          continue

      avg = len(f) / f.samplerate / len(text)
      if avg < mi or avg > mx:
          print("Did skip")
          continue

      new_file.write(wav + "|" + text)
      
      
