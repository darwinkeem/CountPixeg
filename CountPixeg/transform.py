import torchvision.transforms as transforms
from cv2 import cv2
from PIL import Image


def preprocessing(image, mask):
    image = image / 255.0
    image = cv2.resize(image, dsize=(256,256),interpolation=cv2.INTER_AREA)
    # mask = cv2.resize(mask, dsize=(32,64),interpolation=cv2.INTER_AREA)
    # image, mask = Image.fromarray(image), Image.fromarray(mask)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # image_transform.transforms.insert(0, RandAugment(2, 14))
    
    return image_transform(image).float(), image_transform(image).float()
