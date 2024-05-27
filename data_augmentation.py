import albumentations as A
import cv2




def augmentation_data(**kwargs):
    transform = A.Compose(
        [   
            #kullanılmıyor.
            A.augmentations.geometric.rotate.Rotate(p=0.3, limit=5, border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(rotate_limit=15, shift_limit=0.0625, scale_limit=0.1, p=0.1),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.augmentations.geometric.rotate.Rotate(p=0.2, limit=179, border_mode=cv2.BORDER_CONSTANT)



        ],
        additional_targets={'image0': 'image'}
    )

    noise = A.Compose(
        [
            A.GaussNoise(var_limit=(20, 50), p=0.3),  # Gauss gürültüsü ekleme
            A.RandomToneCurve(scale=0.1,p=0.1),
            # A.Blur(blur_limit=1, p=0.2),
            #A.CLAHE(clip_limit=4.0, tile_grid_size=(16, 16),p=1), # piksel bazlı histogram eşitleme
            A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, p=0.3)

        ]
    )

    return transform,noise

