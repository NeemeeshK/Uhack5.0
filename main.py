import  cv2
import random 
import schedule
from PersonDetection.person_and_phone import person_and_cell_count
from deepface import DeepFace
from head_detection.head_pose_estimation import head_pose
import time
import numpy as np



def get_emotion(img_cropped):
    try:
        result = DeepFace.analyze(img_cropped,actions=['emotion'],detector_backend='mtcnn')
        if pred in ['happy','neutral','surprise']:
            pred='positive'
        else:
            pred='negative'
        if result['dominant_emotion'] in ['happy','neutral','surprise']:
            return 1
        elif result['dominant_emotion'] in ['sad','angry','fear','disgust']:
            return 0
        else:
            return 0.5
    except:
        return 0.5

def crop_person(person, frame):
        x, y, w, h = tuple(map(int,(person[0].numpy())*np.array([frame.shape[1],frame.shape[0],frame.shape[1],frame.shape[0]])))
        x1,y1,x2,y2=x,y,x+w,y+h
        p1,p2=(x1-int(w/10),y1-int(h/10)),(x2+int(w/10),y2+int(h/10))
        if p1[0]<0:
            p1=(0,p1[1])
        if p1[1]<0:
            p1=(p1[0],0)
        if p2[0]>frame.shape[1]:
            p2=(frame.shape[1],p2[1])
        if p2[1]>frame.shape[0]:
            p2=(p2[0],frame.shape[0])
        return frame[p1[1]:p2[1],p1[0]:p2[0]]

def main():
    camera = cv2.VideoCapture(r"C:\Users\nimis_r\OneDrive\Desktop\UHACK\IMG_3413.MOV")
    counts = [(0,0,0)]
    n=0
    A_index=[]
    while True:
        ret, frame = camera.read()
        a , b, list_persons = person_and_cell_count(frame)
        c=0
        d=0
        for i in list_persons:
            cropped_image = crop_person(i, frame)
            
            # Expression Recognition on Cropped Image
            c+= get_emotion(cropped_image)
            
            # Head Pose Estimation on Cropped Image
            d+= head_pose(cropped_image)
        n=max(n,a)
        A_index.append((((n-a)/n)+(b/n)+(c/n)+(d/n))/4)
        print(A_index)
        print("Attentivity Index: ",np.mean(A_index))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(5)
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()