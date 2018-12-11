from imageai.Detection import ObjectDetection
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-r", "--result", required=True, help="Path to the output image")

args = vars(ap.parse_args())
execution_path = os.getcwd()

#resnet50_coco_best_v2.0.1.h5
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "CNN_models/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(airplane=True, car=True, truck=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , args["image"]), output_image_path=os.path.join(execution_path , args["result"]))

for eachObject in detections:
    if(eachObject["name"] == 'truck'):
        eachObject["name"] = 'car'
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )