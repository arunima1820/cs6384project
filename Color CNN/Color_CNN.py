from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split #To load datasets from PyTorch
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from torchvision.datasets import ImageFolder
from DogBreedDataset import DogBreedDataset
import matplotlib.pyplot as plt

class DogColorCNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), #Input: 224x224x32
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #Output: 112x112x32
            
            nn.Conv2d(32, 64, 3, 1, 1), #Input: 112x112x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #Output: 56x56x128
            
            nn.Conv2d(128, 256, 3, 1, 1), #Input: 56x56x256
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #Output: 28x28x256
            
            nn.Conv2d(256, 256, 3, 1, 1), 
            nn.ReLU(),            
            nn.MaxPool2d(2, 2), #Output: 14x14x256
            
            nn.Conv2d(256, 256, 3, 1, 1), 
            nn.ReLU(),            
            nn.MaxPool2d(2, 2), #Output: 7x7x256            
            
            nn.Flatten(),
            nn.Linear(7*7*256, 512),  #16384, 512),
            nn.ReLU(),
            nn.Linear(512, 120), #For 120 breeds
            nn.LogSoftmax(dim=1)
        )    
        """Options when mat shape mismatch: 1. Ensure input to layer has right shape 2. Change in_features in linear layer.
           Now, "TypeError: cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not NoneType"
           where y_hat has none. I'm assuming my input features is correct w.r.t. input image size. Not sure why
           y_hat is none and not a tensor.
        """
    
    def forward(self, x):
        self.model(x)

def train(ds):
    for epoch in range(10): #training in 10 epochs
        for batch in ds:
            X, y = batch
            X, y = X.to('cpu'), y.to('cpu')
            
            y_hat = cnn_color_model(X)
            print(f"y_hat: {y_hat}")
            
            loss = loss_fn(y_hat, y)
            
            #Perform backpropagation
            optimizer.zero_grad() #Zero out any existing gradients
            loss.backward() #backpropagation
            optimizer.step() #Gradient descent
            
        print(f"Epoch: {epoch}, loss: {loss.item()}")
    
    with open('model_state.pt', 'wb') as f:
        save(cnn_color_model.state_dict(), f)
        
def createDatasets():
    #Split dataset into training, validation, and test datasets
    test_percent = 0.3
    train_percent = 0.6
    val_percent = 0.1
    
    train_size = int(len(dataset) * train_percent)
    val_size = int(len(dataset) * val_percent)
    test_size = int(len(dataset) * test_percent)
    
    #ds sizes: 12348, 2058, 6174    
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])      
    
    #Create transforms for each dataset
    train_transform = transforms.Compose([
        transforms.Resize((224,224)), #(256,256)),
        #transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
        #transforms.RandomHorizontalFlip(p=0.3),
        #transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
    ])  
    
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),        
        transforms.ToTensor(),
    ]) 
    
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),        
        transforms.ToTensor(),
    ]) 
    
    train_dataset = DogBreedDataset(train_data, train_transform)
    val_dataset = DogBreedDataset(val_data, val_transform)
    test_dataset = DogBreedDataset(test_data, test_transform)
    
    return (train_dataset, val_dataset, test_dataset)

def renameBreed(name):
    return ' '.join(' '.join(name.split('-')[1:]).split('_'))

cnn_color_model = DogColorCNNClassifier().to('cpu')
optimizer = Adam(cnn_color_model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

if __name__ == '__main__':
    #Get dataset
    dataset = ImageFolder('data/Images') #, ToTensor())
    
    #Establish breed classes (and rename them)
    breeds = []
    
    for breed in dataset.classes:
        breeds.append(renameBreed(breed))
        
    #Split dataset into training, validation, and test datasets
    test_percent = 0.3
    train_percent = 0.6
    val_percent = 0.1
    
    train_size = int(len(dataset) * train_percent)
    val_size = int(len(dataset) * val_percent)
    test_size = int(len(dataset) * test_percent)
    
    #ds sizes: 12348, 2058, 6174    
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])  
                 
    #Get datasets
    train_dataset, val_dataset, test_dataset = createDatasets()
    
    #Get dataloaders
    train_dataset = DataLoader(train_dataset, 32)
    
    """
    for i in range(10,25):
        img, label = train_dataset[i]
        print(dataset.classes[label])  
        img = img.permute(1,2,0)  
        plt.imshow(img)
        plt.show()
        print(type(img))
    """
    #Need to figure out a good way to make the images the same size without too much work
    train(train_dataset)
    
    
    