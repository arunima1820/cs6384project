from torch.utils.data import Dataset

class DogBreedDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        img, label = self.ds[index]
        
        if self.transform:
            img = self.transform(img)
            return img, label