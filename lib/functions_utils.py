
import pandas as pd
import numpy as np
import glob 
import rasterio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import laspy
#Data manipulation 


def rep_df_train_test_val(data_df, repartition, random_seed):
    """
    Splits the dataframe into training, validation, and test sets based on species.
    
    Parameters:
    - data_df: DataFrame containing the data with a 'species' column.
    - repartition: Dictionary with keys 'train', 'val', 'test' and their corresponding fractions.
    - random_seed: Seed for reproducibility.
    
    Returns:
    - data_df: DataFrame with an additional 'dataset' column indicating the split.
    """
    data_df["dataset"] = "None"
    for species in data_df['species'].unique():
        specie_df = data_df[data_df['species'] == species]
        specie_df_train = specie_df.sample(frac=repartition['train'], random_state=random_seed)
        specie_df_remaining = specie_df.drop(specie_df_train.index)
        specie_df_val = specie_df_remaining.sample(frac=repartition['val']/(repartition['val'] + repartition['test']), random_state=42)
        specie_df_test = specie_df_remaining.drop(specie_df_val.index)
        
        data_df.loc[specie_df_train.index, 'dataset'] = 'train'
        data_df.loc[specie_df_val.index, 'dataset'] = 'val'
        data_df.loc[specie_df_test.index, 'dataset'] = 'test'
    return data_df

def extrating_data_to_df(DATA_dir,repartition = { 'train': 0.8, 'val': 0.1, 'test': 0.1}, random_seed = 42, new_rep = False):
    print("Getting all the files .tiff and .laz path...")
    aerial_files = glob.glob(DATA_dir + "**/**/**/*.tiff")
    lidar_files = glob.glob(DATA_dir + "**/**/**/*.laz")
    #Sort the files to ensure the matching between aerial and lidar files
    aerial_files.sort()
    lidar_files.sort()

    files_path = aerial_files + lidar_files
    print(f"Number of files : {len(aerial_files)+len(lidar_files)}")


    #Creating a dataframe with the files path and extracting metadata from the path
    data_df = pd.DataFrame(aerial_files, columns=['aerial_path'])
    data_df["lidar_path"] = lidar_files
    data_df["species"] = data_df["aerial_path"].apply(lambda x: x.split("/")[-4])
    data_df["type"] = data_df["aerial_path"].apply(lambda x: x.split("/")[-3])
    if  new_rep:
        data_df = rep_df_train_test_val(data_df, repartition = repartition, random_seed = random_seed)
    else:
        data_df["dataset"] = data_df["aerial_path"].apply(lambda x: x.split("/")[-2])

    return data_df



def detailed_distribution(df):
    print("=" * 50)
    print("🌳 Sample by species:")
    species_counts = df['species'].value_counts()
    species_pct = (species_counts / len(df) * 100).round(1)
    for species, count in species_counts.items():
        pct = species_pct[species]
        print(f"  {species}: {count} samples ({pct}%)")
    
    print("\n" + "=" * 50)
   
    print("🎯 Dataset distribution:")
    dataset_counts = df['dataset'].value_counts()
    dataset_pct = (dataset_counts / len(df) * 100).round(1)
    for dataset, count in dataset_counts.items():
        pct = dataset_pct[dataset]
        print(f"  {dataset}: {count} samples ({pct}%)")
    
    print("\n" + "=" * 50)
    # Vérification des ratios attendus
    print("\n🔍 Detailed distribution:")
    pct_by_species = pd.crosstab(df['species'], df['dataset'], normalize='index') * 100
    expected = {'train': 80, 'val': 10, 'test': 10}  # ratios attendus
    
    for species in df['species'].unique():
        print(f"\n🌳 {species}:")
        species_pct = pct_by_species.loc[species]
        
        for dataset in ['train', 'val', 'test']:
            if dataset in species_pct.index:
                actual = species_pct[dataset]
                expected_val = expected[dataset]
                diff = abs(actual - expected_val)
                
                status = "✅" if diff < 2 else "⚠️" if diff < 5 else "❌"
                print(f"  {dataset:5}: {actual:5.1f}% (attendu: {expected_val}%) {status}")




def create_dataloader(df,batch_size = 4, input_type = "imagery", transform=None):
    """
    Create dataloaders for training, validation and testing.

    Args:
        df (pd.DataFrame): DataFrame containing file paths and labels.
        batch_size (int): Batch size for the dataloaders.
        input_type (str): Type of input data ('aerial', 'lidar', 'fusion').
    Returns:
        dict: dataloader
    """
    #creating dataset 
    dataset = TreeDataset(df, input_type=input_type, transform = transform)

    #creating dataloaders
    dataloader = DataLoader(
                dataset, 
                batch_size = batch_size, 
                shuffle=True,
                num_workers=0
            )

    return dataloader


class TreeDataset(torch.utils.data.Dataset):
    def __init__(self, df, input_type='imagery', transform=None, target_points=100000):
        self.df = df
        self.input_type = input_type
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(df['species'].unique())}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        self.target_points = target_points
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.class_to_idx[row['species']]
        
        # Load aerial image
        if self.input_type in ['imagery', 'fusion']:
            inputs = self.load_aerial_image(row["aerial_path"])
        else:
            inputs = self.load_lidar_data(row["lidar_path"])
        
        if self.transform is not None:
            inputs = self.transform(inputs)

        label = self.class_to_idx[row["species"]]

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def load_aerial_image(self, path):
        with rasterio.open(path) as src:
            img = src.read()  # Lire toutes les bandes
            #resize to 4*128*128
            img = np.resize(img, (4, 128, 128))
            img = np.transpose(img, (0, 1, 2))  # Convertir en HWC
            img = img / 255.0  # Normalisation
        return img
    def normalize_point_cloud(self,points, target_points=1024):
    
        if len(points) > target_points:
            # Random sub-sampling
            indices = np.random.choice(len(points), target_points, replace=False)
            return points[indices]
        
        elif len(points) < target_points:
            # Random over-sampling
            indices = np.random.choice(len(points), target_points, replace=True)
            return points[indices]
        
        return points 

    def load_lidar_data(self, path):
        with laspy.open(path) as fh:
            las = fh.read()
        output = np.column_stack([las.x, las.y, las.z])
        output = self.normalize_point_cloud(output, target_points = self.target_points)
        output = np.transpose(output)  # Convertir en (C, N)
        return output
    
    # Visualization and evaluation 
def getting_metrics(results,EPOCHS):
    avg_training_accuracy, avg_validation_accuracy =[], []
    avg_training_loss, avg_validation_loss=[],[]
    lr = []
    for result in results:
        avg_training_accuracy.append(result['avg_train_acc'])
        avg_validation_accuracy.append(result['avg_val_acc'])
        avg_validation_loss.append(result['avg_valid_loss'])
        avg_training_loss.append(result['avg_train_loss'])
        lr = np.concatenate((lr,result['lrs']))
    epoch_count=[]
    for i in range(1,EPOCHS+1):
        epoch_count.append(i)
    return avg_training_accuracy, avg_validation_accuracy, avg_training_loss, avg_validation_loss, lr, epoch_count

def plot_metric(train_metric,test_metric,label,epoch_count,option_label = "",color = "orange"):
  plt.title(f"{label} per epoch")
  plt.plot(epoch_count,train_metric,label=f"Training {label} {option_label}",linewidth = '1',color = color)
  plt.plot(epoch_count,test_metric,"--",label=f"Test {label} {option_label}",linewidth = '1',color = color)
  plt.xlabel('Epoch')
  plt.ylabel(f'{label} per epoch')
  plt.legend()
  
def plot_all_metrics(tr_acc,ts_acc,tr_loss,ts_loss,lr,epoch_count,label,color="orange"):
  #Visualize metrics over epochs

  plt.subplot(1,3,1)
  plot_metric(tr_loss,ts_loss,"Loss",epoch_count,option_label = label,color = color)
  plt.subplot(1,3,2)
  plot_metric(tr_acc,ts_acc,"Accuracy",epoch_count,option_label = label, color =color)

  #Visualize lr evolution
  plt.subplot(1,3,3)
  epoch = epoch_count
  plt.plot(epoch,lr)
  plt.title("learning_rate over epochs")
  plt.xlabel('Epoch')
  plt.ylabel('Learning rate')
