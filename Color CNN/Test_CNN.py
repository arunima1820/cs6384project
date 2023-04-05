import torch
from Color_CNN import DogColorCNNClassifier
from torch import load
from PIL import Image
from torchvision.transforms import ToTensor

if __name__ == "__main__":
    clf = DogColorCNNClassifier().to('cpu')
    
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))
        
    #Load in test image
    img_name = "img_1.jpg"
    img1 = Image.open(img_name)
    img_tensor = ToTensor()(img1).unsqueeze(0).to('cpu') #unsqueeze bc we want to parse through a single sample
    
    print(torch.argmax(clf(img_tensor)))