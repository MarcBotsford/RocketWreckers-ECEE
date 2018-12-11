from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
#resnet50_coco_best_v2.0.1.h5
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "CNN_models/yolo.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(airplane=True, car=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "jpegs/00000137.jpg"), output_image_path=os.path.join(execution_path , "test_images/sat_test_2.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )