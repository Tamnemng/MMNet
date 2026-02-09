import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class Feeder(Dataset):
    def __init__(self, label_path, rgb_path, 
                 temporal_rgb_frames=5, 
                 random_flip=False, 
                 random_crop=False, 
                 transform=None,
                 debug=False,    # <--- Thêm tham số này để hứng giá trị từ processor
                 **kwargs):      # <--- Thêm kwargs để hứng bất kỳ tham số thừa nào khác
        
        self.label_path = label_path
        self.rgb_path = rgb_path
        self.temporal_rgb_frames = temporal_rgb_frames
        self.random_flip = random_flip
        self.debug = debug # Lưu lại nếu cần dùng (hiện tại thì chưa cần)

        self.load_data()
        
        # Transform cho ResNet (224x224)
        if self.train_val == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def load_data(self):
        # Logic kiểm tra xem đang load tập train hay tập val dựa vào chuỗi trong config
        if 'val' in self.label_path or 'test' in self.label_path:
            self.train_val = 'val'
            # --- BẠN ĐIỀN DỮ LIỆU TEST/VAL VÀO DƯỚI ĐÂY ---
            self.data_dict = [
                # Ví dụ:
                # {"file_name": "a05_s04_e02_v03", "label": 5}, 
                
            ]
        else:
            self.train_val = 'train'
            # --- BẠN ĐIỀN DỮ LIỆU TRAIN VÀO DƯỚI ĐÂY ---
            self.data_dict = [
                # Ví dụ:
                # {"file_name": "a01_s01_e01_v01", "label": 0},
                
            ]

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        # Lấy thông tin từ data_dict
        info = self.data_dict[index]
        filename = info['file_name']
        label = int(info['label'])
        
        # Đường dẫn ảnh
        img_path = self.rgb_path + filename + '.png'
        
        try:
            rgb = Image.open(img_path).convert('RGB')
        except:
            print(f"Error loading image: {img_path}")
            # Tạo ảnh đen tạm nếu lỗi để không dừng chương trình
            rgb = Image.new('RGB', (224, 224))

        # Data Augmentation (chỉ cho tập train nếu cần)
        if self.train_val == 'train' and self.random_flip and np.random.random() < 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)

        # Apply transform (Resize, Normalize)
        rgb = self.transform(rgb)
        
        # Trả về: Ảnh, Nhãn, Tên file (để debug)
        return rgb, label, filename