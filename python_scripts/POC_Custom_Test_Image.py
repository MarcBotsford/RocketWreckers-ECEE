from imageai.Prediction.Custom import CustomImagePrediction
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")

args = vars(ap.parse_args())
execution_path = os.getcwd()
#resnet_model_ex-020_acc-0.651714.h5
prediction = CustomImagePrediction()
prediction.setModelTypeAsSqueezeNet()
prediction.setModelPath(os.path.join(execution_path, "CNN_models/model_ex-100_acc-0.817708.h5"))
prediction.setJsonPath(os.path.join(execution_path, "ImageAI_Custom_CNN/vehicles/json/model_class.json"))
prediction.loadModel(num_objects=3)

predictions, probabilities = prediction.predictImage(os.path.join(execution_path,args["image"]) , result_count=2)


for eachPrediction, eachProbability in zip(predictions, probabilities):
    if(eachPrediction == "truck"):
        eachPrediction == "car"
    print(eachPrediction , " : " , eachProbability)