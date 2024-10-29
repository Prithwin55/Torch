import cv2 as cv
import numpy as np
from keras_facenet import FaceNet
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import LabelEncoder
import joblib
import torch

embedder = FaceNet()

class FastMTCNN:
    def __init__(self, stride=1, resize=1, *args, **kwargs):
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs).eval()

    def __call__(self, frames):
        if self.resize != 1:
            frames = [
                cv.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]
        boxes, _ = self.mtcnn.detect(frames[::self.stride])
        return boxes

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fast_mtcnn = FastMTCNN(stride=4, resize=1, margin=14, factor=0.6, keep_all=True, device=device)

data = np.load('faces_embeddings.npz')
X_embedding = data['arr_0']
y_labels = data['arr_1']
svc_model = joblib.load('svm_model')

label_encoder = LabelEncoder()
label_encoder.fit(y_labels)

def get_embedding(face_img):
    if face_img is None or face_img.size == 0:
        print("Warning: face_img is empty or invalid.")
        return None
    
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    embedding = embedder.embeddings(face_img)
    return embedding[0]

def recognize_faces_realtime():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames = [frame_rgb]

        boxes = fast_mtcnn(frames)

        if boxes is not None and boxes[0] is not None:
            for box in boxes[0]:
                if box is not None:
                    x1, y1, x2, y2 = [int(b) for b in box]

                    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                        print("Invalid bounding box coordinates. Skipping this face.")
                        continue

                    face_img = frame_rgb[y1:y2, x1:x2]
                    if face_img.size == 0:
                        print("Invalid face image. Skipping.")
                        continue

                    face_embedding = get_embedding(face_img)
                    if face_embedding is None:
                        continue
                    
                    face_embedding = face_embedding.reshape(1, -1)
                    
                    prediction = svc_model.predict(face_embedding)
                    prob = svc_model.predict_proba(face_embedding)
                    label = label_encoder.inverse_transform(prediction)[0]
                    confidence = prob[0][prediction[0]] * 100
                    
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.putText(frame, f"{label} ({confidence:.2f}%)", (x1, y1 - 10), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv.imshow('Face Recognition', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

recognize_faces_realtime()
