
import pandas as pd
import numpy as np
import glob 
import rasterio
#Data manipulation 

def extrating_data_to_df(DATA_dir):
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
    return data_df

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




import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import laspy
import os

class RGBLidarDataset(Dataset):
    """Creation of dataloader for lidar and Vis-NIR aerial images for tree species classification"""
    
    def __init__(self, df, mode='train', input_type ='aerial', transform=None, target_size=(224, 224)):
        """
        Args:
            df: DataFrame with colomns ['rgb_path', 'lidar_path', 'species', ...]
            mode: 'train', 'val', 'test'
            input_type: 'lidar', 'aerial', 'fusion'
            transform: transform to apply
            target_size: target size for images and heightmaps
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.input_type = input_type
        self.target_size = target_size
        
        # Transform step (augmentation only for training)
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
            
        # Mapping of classes 
        self.class_to_idx = {cls: idx for idx, cls in enumerate(df['species'].unique())}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        print(f"Dataset of {len(self.df)} samples, divised in {self.num_classes} classes ({list(self.class_to_idx.keys())})")
        print(f"Input type: {input_type}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Charge un échantillon"""
        
        try:
            row = self.df.iloc[idx]
            
            # 1. CHARGEMENT Vis+NIR
            aerial_image = None
            if self.input_type in ['fusion', 'aerial']:
                aerial_image = self._load_aerial(row['aerial_path'])

            # 2. CHARGEMENT LIDAR
            lidar_data = None
            if self.input_type in ['fusion', 'lidar']:
                lidar_data = self._load_lidar(row['lidar_path'])
            
            # 3. FUSION/PRÉPARATION
            input_data = self._prepare_input(aerial_image, lidar_data)

            # 4. LABEL
            label = self.class_to_idx[row['species']]
            
            # 5. TRANSFORMATIONS
            if self.transform:
                input_data = self.transform(input_data)
            
            return input_data, label
            
        except Exception as e:
            print(f"❌ Erreur chargement échantillon {idx}: {e}")
            # If error a random sample is returned
            return self.__getitem__(np.random.randint(0, len(self.df)))
    
    def _load_aerial(self, rgb_path):
        """Charge aerial image """
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"aerial file not found: {rgb_path}")
        with rasterio.open(rgb_path) as src:
           img = src.read()
           img = np.moveaxis(img, 0, -1) #switch channel order
        return np.array(img)
    
    def _load_lidar(self, lidar_path):
        """Charge données LiDAR et crée heightmap"""
        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
        
        # Loading LAZ
        with laspy.open(lidar_path) as fh:
            las = fh.read()
        
        # Heightmap Creation
        heightmap = self._create_heightmap(las.x, las.y, las.z, las)
        return heightmap
    
    def _create_heightmap(self, x, y, z, las):
        """CHECK THAT FUNCTIONS"""
        
        H, W = self.target_size
        
        # Projection grid 
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Avoid division by zero
        if (x_max - x_min) == 0 or (y_max - y_min) == 0:
            return np.zeros((H, W, 4), dtype=np.float32)
        
        # Grid indixes
        x_indices = ((x - x_min) / (x_max - x_min) * (W - 1)).astype(int)
        y_indices = ((y - y_min) / (y_max - y_min) * (H - 1)).astype(int)
        
        # Clip pour éviter out of bounds
        x_indices = np.clip(x_indices, 0, W - 1)
        y_indices = np.clip(y_indices, 0, H - 1)
        
        # Création des canaux
        heightmap = np.zeros((H, W, 4), dtype=np.float32)
        
        # Canal 0: Hauteur maximale
        for i in range(len(x)):
            row, col = y_indices[i], x_indices[i]
            heightmap[row, col, 0] = max(heightmap[row, col, 0], z[i])
        
        # Canal 1: Densité (nombre de points)
        for i in range(len(x)):
            row, col = y_indices[i], x_indices[i]
            heightmap[row, col, 1] += 1
        
        # Canal 2: Intensité moyenne
        if hasattr(las, 'intensity'):
            intensity_sum = np.zeros((H, W))
            for i in range(len(x)):
                row, col = y_indices[i], x_indices[i]
                intensity_sum[row, col] += las.intensity[i]
            
            # Moyenne (évite division par zéro)
            mask = heightmap[:, :, 1] > 0
            heightmap[mask, 2] = intensity_sum[mask] / heightmap[mask, 1]
        
        # Canal 3: Ratio premier/dernier retour
        first_returns = np.zeros((H, W))
        last_returns = np.zeros((H, W))
        
        for i in range(len(x)):
            row, col = y_indices[i], x_indices[i]
            if las.return_number[i] == 1:
                first_returns[row, col] += 1
            if las.return_number[i] == las.number_of_returns[i]:
                last_returns[row, col] += 1
        
        # Ratio (évite division par zéro)
        mask = last_returns > 0
        heightmap[mask, 3] = first_returns[mask] / last_returns[mask]
        
        return heightmap
    
    def _prepare_input(self, aerial_image, lidar_data):
        """Prépare input selon méthode de fusion"""
        
        if self.input_type == 'fusion':
            # Fusion 
            if aerial_image is None or lidar_data is None:
                raise ValueError("Fusion need Lidar and Aerial data")
            
            # RGB: (H, W, 3) + LiDAR: (H, W, 4) → (H, W, 7)
            fused = np.concatenate([aerial_image, lidar_data], axis=-1)
            return fused.astype(np.float32) / 255.0  # Normalisation

        elif self.input_type == 'aerial':
            return aerial_image.astype(np.float32) / 255.0

        elif self.input_type == 'lidar':
            return lidar_data.astype(np.float32)
        
        else:
            raise ValueError(f"Méthode fusion inconnue: {self.fusion_method}")
    
    def _get_default_transforms(self):
        """Transformations par défaut selon le mode"""

        
        
        # Transformations pour rgb et lidar
        if self.mode == "train":
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
            ])
    
    def get_class_weights(self):
        """Compute class weights for balanced loss"""
        class_counts = self.df['species'].value_counts()
        total_samples = len(self.df)
        
        weights = []
        for cls in self.class_to_idx.keys():
            weight = total_samples / (len(self.class_to_idx) * class_counts[cls])
            weights.append(weight)
        
        return torch.FloatTensor(weights)