import torch
import torch.utils.data.dataloader
import torchvision.transforms as tr
import numpy as np

import PIL.Image as Image
# data = np.load('classes.npz')
# classes = data['arr_0']

classes=[
    "alouatta_palliata",
    "aotus_nigriceps",
    "cacajao_calvus",
    "cebuella_pygmea",
    "cebus_capucinus",
    "erythrocebus_patas",
    "macaca_fuscata",
    "mico_argentatus",
    "saimiri_sciureus",
    "trachypithecus_johnii"
]

model=torch.load("Model.pth",weights_only=False)

image_transforms=tr.Compose([
    tr.Resize((224,224)),
    tr.ToTensor()
])

def classify(model,image_transforms,image_path,classes):
    model=model.eval()
    image=Image.open(image_path)
    image=image_transforms(image).float()
    image=image.unsqueeze(0)

    
    output=model(image)

    _,predicted=torch.max(output.data,1)
    print(predicted[0])

    result=classes[predicted.item()]

    print(result)


classify(model,image_transforms,"download.jpeg",classes)

#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