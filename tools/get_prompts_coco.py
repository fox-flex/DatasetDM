classes = {
        0: 'background',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        12: 'stop sign',
        13: 'parking meter',
        14: 'bench',
        15: 'bird',
        16: 'cat',
        17: 'dog',
        18: 'horse',
        19: 'sheep',
        20: 'cow',
        21: 'elephant',
        22: 'bear',
        23: 'zebra',
        24: 'giraffe',
        25: 'backpack',
        26: 'umbrella',
        27: 'handbag',
        28: 'tie',
        29: 'suitcase',
        30: 'frisbee',
        31: 'skis',
        32: 'snowboard',
        33: 'sports ball',
        34: 'kite',
        35: 'baseball bat',
        36: 'baseball glove',
        37: 'skateboard',
        38: 'surfboard',
        39: 'tennis racket',
        40: 'bottle',
        41: 'wine glass',
        42: 'cup',
        43: 'fork',
        44: 'knife',
        45: 'spoon',
        46: 'bowl',
        47: 'banana',
        48: 'apple',
        49: 'sandwich',
        50: 'orange',
        51: 'broccoli',
        52: 'carrot',
        53: 'hot dog',
        54: 'pizza',
        55: 'donut',
        56: 'cake',
        57: 'chair',
        58: 'couch',
        59: 'potted plant',
        60: 'bed',
        61: 'dining table',
        62: 'toilet',
        63: 'tv',
        64: 'laptop',
        65: 'computer mouse',
        66: 'remote',
        67: 'keyboard',
        68: 'cell phone',69: 'microwave',70: 'oven',71: 'toaster',72: 'sink',73: 'refrigerator',
        74: 'book',
        75: 'clock',
        76: 'vase',
        77: 'scissors',
        78: 'teddy bear',
        79: 'hair drier',
        80: 'toothbrush'

    }

prompts_list = ["photo of a {} is walking",
                    "photo of a {} is eating",
                    "photo of a {} is play",  
                    
                    #sports
                    "photo of a {} is dancing",
                    "photo of a {} is doing Yoga","photo of a {} is doing Fitness",
                    
                    "photo of a {} is doing High jumping","photo of a {} is Cycling",
                    "photo of a {} is running","photo of a {} is Fishing",
                    "photo of a {} is doing Judo","photo of a {} is Climbing",
                    
                    # scenario
                    "photo of a {} is walking in the street","photo of a {} is in the road",
                    "photo of a {} is playing at home",
                    "photo of a {} is in on the mountain","photo of a {} is in on the mountain",
                    "photo of a {} is crossing a road","photo of a {} is sitting"]
names = ['person',"man","woman","child","boy","girl","old man","teenager"]

import os

def get_prompts(prompt_root):
    for idx in classes:
        
        if idx==0:
            continue
        
        
        class_target = classes[idx]
        
        prompt_path = os.path.join(prompt_root, class_target+".txt")
        sub_classes_list = []
        
        if os.path.exists(prompt_path):
            with open(prompt_path,"r") as f2:
                for line3 in f2.readlines():
                    sub_classes_list.append(line3.strip())
        
        if idx==1:
            sub_classes_list=[]
            for name in names:
                for prompts_line in prompts_list:
                    sub_classes_list.append(prompts_line.format(name))
                            
        prompt_path = os.path.join(prompt_root,"general.txt")
        with open(prompt_path,"r") as f2:
            for line3 in f2.readlines():
                sub_classes_list.append(line3.replace("\n",""))
            
        # print("prompt candidates:",len(sub_classes_list))
        if len(sub_classes_list) == 0:
            # print("No prompts for class:",class_target)
            continue
        print(f"Prompt for class '{class_target}': num: {len(sub_classes_list)}",)
        # continue
        yield sub_classes_list

