import os
from cv2 import cv2

import torch
import torchvision
import torch.nn as nn

from model import U_Net
from UPPNet import UPPNet


MODEL_PATH = './unet/UPPNet_OCR_Check60.pt'
VIDEO_PATH = './video/US_ABC___170_007.mp4'
# CCI_PARAM = 5
THRESHOLD = 0.5

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":
    model = UPPNet(3, 1, ocr=True, check=True)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model.to(device=device)

    first = 1
    cap = cv2.VideoCapture(VIDEO_PATH)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(512, 256))
        frame = torchvision.transforms.ToTensor()(frame)
        frame = frame.unsqueeze(0).to(device)
        gray = model(frame)[0]
        torchvision.utils.save_image(gray, './temp/ohpred.png')
        gray = cv2.imread('./temp/ohpred.png')
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = ~gray

        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)

        contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            cv2.drawContours(gray, [contours[i]], 0, (0, 0, 255), 2)
            prev_size = []
            print(cv2.contourArea(contours[i]))
            cv2.putText(gray, 'Candidate '+str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
            print(i, hierachy[0][i])
            cv2.imshow("src", gray)
            if first == 1:
                prev_size.append(cv2.contourArea(contours[i]))
            else:
                if cv2.contourArea(contours[i]) < prev_size[i] * THRESHOLD:
                    cv2.putText(gray, 'Vein '+str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
                if cv2.contourArea(contours[i]) > prev_size[i]:
                    prev_size[i] = cv2.contourArea(contours[i])

            cv2.waitKey(0)
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            first = 0
            break
        
        

    cap.release()
    cv2.destroyAllWindows()
