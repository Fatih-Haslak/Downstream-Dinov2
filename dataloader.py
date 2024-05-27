#new dataloader.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import random
import torch
import cv2
from torch.utils.data import Dataset
random.seed(42) 
import matplotlib.pyplot as plt


class ValfDataSetLoader(Dataset):
    def __init__(self,root_dir,transform=None,noise=None,batch_size=1):
        self.root_dir=root_dir #Train or Validation Root Folders r/Dataset/Train/*
        self.batch_size=batch_size
        self.transform=transform
        self.noise=noise
        self.path_list=[]
        #Klasörlerin içine gir ve tüm png uzantılı dosyaları oku, ardından bunları bir listede tut ve tüm veriler elimizde.
        for folders in glob.glob(os.path.join(self.root_dir, '*.png')):
            self.path_list.append(folders)
     
        random.shuffle(self.path_list) #Klasörleri tek tek dolastigi icin veriler sirali geliyordu. Bundan dolayı shuffle ediyoruz.
    def __len__(self):
        #Her bir epoch'da egitime girecek olan veri sayisi, counteri etkileyecegi icin //batch_size islemini uyguladik.
        return len(self.path_list) // self.batch_size 



    def __getitem__(self, counter):
        #Alınacak Verilerin baslangicinin indexi
        start_batch_index= counter * self.batch_size
        #Alınacak Verilerin sonunun indexi  "Her bir Batch icin"
        end_batch_index= start_batch_index + self.batch_size
        
        #Her döngüde alinacak olan batchlerin pathlari "List Veri Tipi"
        batch_img_paths=self.path_list[start_batch_index:end_batch_index]
        
        #batchleri tutacak olan listeler
        image_batch=[] # en sonunda [batch_size,channel,witdh,height]
        label_batch=[] # [batch_size]

        for paths in batch_img_paths:
     
            img=cv2.imread(paths)
            img=cv2.resize(img,(280, 280),interpolation=cv2.INTER_LINEAR)
            # img = cv2.bilateralFilter(img,6,25,25)

            label_str=paths.split("\\")[2]
  
            if(label_str=="NOK"):
                label=0
            elif(label_str=="OK"):
                label=1
     

            label_tensor=torch.tensor(label).long()

            #transform
            if(self.transform is not None):
                aug_imgs = self.transform(image=img)
                second_aug_images = self.noise(image = aug_imgs['image'])['image']
                #second_aug_images[second_aug_images<=10]=0 #change
                second_aug_images=second_aug_images/255 #float32 donusumu ıcın gereklı yoksa sıyah beyaz olur verin [0,1]
                # plt.imshow(second_aug_images)
                # plt.show()
                img_tensor=torch.from_numpy(second_aug_images).unsqueeze(0).permute(0,3,1,2).float()
            #validation
            else:
                img=img/255
                img_tensor=torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).float()

    
            # print(img_tensor.shape)
            # önce listeye attik ve sonradan cat ederek istenilen boyutlandırmayı yapmaya calistik
            image_batch.append(img_tensor)
            label_batch.append(label_tensor)

        
        if not image_batch:
           return image_batch[0].float(), label_batch[0].view(-1, 1).float()
        else:
            transformed_image = torch.cat(image_batch, dim=0)
            transformed_label = torch.tensor(label_batch).view(-1, 1).float()            
            return transformed_image, transformed_label
           

