
# Pythono3 code to rename multiple  
# files in a directory or folder 
  
# importing os module 
import os 
execution_path = os.getcwd()
i = 1
for filename in os.listdir("ImageAI_Custom_CNN/vehicles/train/nothing/nothing-train-images"):
    index = str(i)
    os.rename("ImageAI_Custom_CNN/vehicles/train/nothing/nothing-train-images/"+filename, "ImageAI_Custom_CNN/vehicles/train/nothing/nothing-train-images-tf/" + str(0) + "_training_" + index + ".jpg")
    i = i + 1 