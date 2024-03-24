import torch
import seaborn as sns 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
import csv

class CustomDataset(Dataset):
    """
    Custom Dataset class for loading data with predefined transformations and labels from a CSV file.
    
    Args:
        data_dir (str): Path to the directory containing the data.
        csv_file (str): Path to the CSV file containing labels for each image.
    """
    def __init__(self, data_dir, csv_file, path_save_dist="label_distribution.png"):
        """
        Initialize the dataset by reading labels from a CSV file.
        Define transformations inside the class.
        """
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize(520),
            transforms.CenterCrop(518),  # Should be a multiple of the model's patch size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
        ])
        
        # Read image names and labels from the CSV file
        self.labels_dict = {}
        with open(csv_file, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader, None)  # Skip the header if there is one
            for row in reader:
                # Assuming first column is image name and second column is label
                self.labels_dict[row[0]] = int(row[1])  # Convert label to integer if necessary
        
        # remove images that do not exist
        for key in list(self.labels_dict.keys()):
            if not os.path.exists(os.path.join(self.data_dir, key + '.jpeg')):
                del self.labels_dict[key]
        
        # List of image names (keys of the dictionary)
        self.image_list = list(self.labels_dict.keys())
        # self.generate_distribution_chart(path_save_dist)

    
    def generate_distribution_chart(self, save_path="label_distribution.png"):
        """
        Generates a bar chart showing the distribution of labels and saves it.

        Args:
            save_path (str, optional): Path where the chart will be saved. 
                                       Defaults to "label_distribution.png".
        """

        # Create a Pandas Series for plotting (includes all labels)
        label_series = pd.Series(self.labels_dict.values())

        # Seaborn bar plot
        sns.countplot(x=label_series)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Distribution of Labels in Dataset")

        # Save the figure
        plt.savefig(save_path) 
        plt.close() 
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Generate one sample of data.
        
        Args:
            idx (int): Index of the sample to return
        
        Returns:
            sample (dict): a sample containing 'image' and 'label'.
        """
        img_name = self.image_list[idx]
        img_path = os.path.join(self.data_dir, img_name + '.jpeg')
        
        # Load the image
        image = Image.open(img_path).convert('RGB')  # Convert to RGB to ensure 3 channels
        
        # Apply the predefined transformations
        image = self.transform(image)
        
        # Retrieve the label for this image
        label = self.labels_dict[img_name]
        
        sample = image, label
        
        return sample
