from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()
#resnet_model_ex-020_acc-0.651714.h5
prediction = CustomImagePrediction()
prediction.setModelTypeAsSqueezeNet()
prediction.setModelPath(os.path.join(execution_path, "CNN_models/SN_No_Planes.h5"))
prediction.setJsonPath(os.path.join(execution_path, "ImageAI_Custom_CNN/vehicles/json/model_class.json"))
prediction.loadModel(num_objects=3)

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "jpegs/plane_image_00031344.jpg"), result_count=5)


for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)