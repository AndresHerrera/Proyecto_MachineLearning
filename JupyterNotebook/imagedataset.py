import yaml
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Path Config
path = "dataset-1.0/"
subpath_annotations = "annotations/"
subpath_images = "images/"

def GetAnnotations():    
    annotation_path = lambda i: path+subpath_annotations+"%03d"%(i)+"_annotation.yaml"
    annotations = [annotation_path(i+1) for i in range(60)]
    annotations_json = []
    for annotation in annotations:
        with open(annotation, 'r') as stream:
            try:
                annotations_json.append(yaml.load(stream))
            except yaml.YAMLError as exc:
                print(exc)
    return annotations_json

# Get all features
boundary = ([15, 90, 20], [110, 230, 80])
lower = np.array(boundary[0],dtype=np.uint8)
upper = np.array(boundary[1],dtype=np.uint8)

def LoadAllDescriptors():
    print("Loading all sift descriptors in dataset from file...")
    all_descriptors = np.load('all_descriptors.npy')
    print("All descriptors loaded ! ")
    return all_descriptors

def GenerateAllDescriptors(annotations):
    print("Extracting all sift descriptors in dataset......")
    all_descriptors = np.arange(128).reshape(1,128)
    for annotation in annotations:
        image_file = annotation['filename']
        img = cv2.imread(path+subpath_images+image_file)
        regions = annotation['annotation']
        for region in regions:
            points = region['points']
            px = points['x']
            py = points['y']
            if(type(px) is list):
                pts = np.array([[int(x),int(y)] for x,y in zip(px,py)], dtype=np.int32)
                mask_region = np.zeros(img.shape, dtype=np.uint8)
                cv2.fillConvexPoly(mask_region, pts, (255,255,255))
                masked_region = cv2.bitwise_and(img,mask_region)
                mask_color = cv2.inRange(masked_region, lower, upper)
                masked_color = cv2.bitwise_and(masked_region, masked_region, mask = mask_color)
                kps , descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(masked_color, None)
                all_descriptors = np.append(all_descriptors, descriptors, axis=0)
            else:
                print('No se pudo generar la region')
    all_descriptors = np.delete(all_descriptors, 0, 0)
    print("All descriptors extracted successfully with shape {}".format(all_descriptors.shape))
    print("Saving all descriptors at all_descriptors.npy")
    np.save('all_descriptors', all_descriptors)
    return all_descriptors

def GenerateFeatures(feature_vector, *args):
    annotations = args[0]
    vector_len = args[1]
    cluster = args[2]
    scaler = args[3]
    y = np.array([])
    X = np.arange(vector_len).reshape(1,vector_len)
    for annotation in annotations:
        image_file = annotation['filename']
        img = cv2.imread(path+subpath_images+image_file)
        regions = annotation['annotation']
        for region in regions:
            points = region['points']
            px = points['x']
            py = points['y']
            if(type(px) is list):
                pts = np.array([[int(x),int(y)] for x,y in zip(px,py)], dtype=np.int32)
                mask_region = np.zeros(img.shape, dtype=np.uint8)
                cv2.fillConvexPoly(mask_region, pts, (255,255,255))
                masked_region = cv2.bitwise_and(img,mask_region)
                mask_color = cv2.inRange(masked_region, lower, upper)
                masked_color = cv2.bitwise_and(masked_region, masked_region, mask = mask_color)
                kps , descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(masked_color, None)
                descriptors = scaler.transform(descriptors)
                X = np.append(X, feature_vector(descriptors, *cluster).reshape(1,vector_len), axis=0)
                y = np.append(y, region['type'])
            else:
                print('No se pudo generar la region in : ' + image_file)
    
    X = np.delete(X, 0, 0)
    # encoding categorical data
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y