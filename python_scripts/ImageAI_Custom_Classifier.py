from imageai.Prediction.Custom import ModelTraining
from imageai.Detection import ObjectDetection
model_trainer = ModelTraining()
model_trainer.setModelTypeAsSqueezeNet()
model_trainer.setDataDirectory("ImageAI_Custom_CNN/vehicles")
model_trainer.trainModel(num_objects=2, num_experiments=100, enhance_data=True, batch_size=16, show_network_summary=True)