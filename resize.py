import pandas as pd
import os

roi_dix = {'x': 330, 'y': 345, 'w': 860, 'h': 720} #ws2
# roi_dix = {'x': 340, 'y': 100, 'w': 980, 'h': 720} # ws3

x, y, w, h = roi_dix['x'], roi_dix['y'], roi_dix['w'], roi_dix['h']

import cv2
import os

def resize_image(data_path, resize_path):
    
    for sub_dir in os.listdir(data_path):

        sub_dir_full_path = os.path.join(data_path, sub_dir)    
        
        for path in os.listdir(sub_dir_full_path):
            path = os.path.join(sub_dir_full_path, path)
            print("Old : ", path)        
            
            new_path = path.replace(data_path, resize_path)

            folder = "/".join(new_path.split("/")[:-1])
            
            if not os.path.isdir(folder):
                os.makedirs(folder)

            print("New : ", new_path)
            
            cap = cv2.VideoCapture(path)

            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                FrameSize=(256, 256)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(new_path,fourcc,fps,(256, 256),True)
            else:
                print("Camera is not opened")  
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"Video : {path} is Finished....\n") 

                    break
                frame = frame[y:y+h, x:x+w]
                rescaled_frame = cv2.resize(frame, (256, 256),interpolation=cv2.INTER_AREA)
                writer.write(rescaled_frame)

            cv2.destroyAllWindows()
            cap.release()
            writer.release()        
            
            
data_path =  'data'
resize_data_path =  'rescaled_data'
resize_image(data_path, resize_data_path)
