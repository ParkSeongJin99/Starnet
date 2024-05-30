import os
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from CustomDataset import CustomDataset  # CustomDataset.py의 CustomDataset 클래스를 import합니다.
from Starnet import StarNet  # Starnet.py의 StarNet 클래스를 import합니다.

def main():
    parser = argparse.ArgumentParser(description="PyTorch StarNet Validation")
    
    parser.add_argument("model", help="directory to load the trained models")
    parser.add_argument("--datapath", metavar="DIR", default="Validation"  ,help="path to dataset")
    args = parser.parse_args()

    # 이미지 데이터셋이 있는 폴더 경로를 지정합니다.
    data_dir = args.datapath
    
    # 데이터셋을 읽어오는 DataLoader를 설정합니다.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = CustomDataset(root_dir=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 사전 훈련된 모델을 로드합니다.
    pretrained_model_path = args.model  # 사전 훈련된 모델의 경로를 지정합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StarNet().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu'))['state_dict'])
    model.eval()
    
    # 각 이미지에 대한 결과값을 계산하고 출력합니다.
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        output = outputs.item()
        label = labels.item()
        error = output - label
        percent_error = (error / label) * 100 if label != 0 else float('inf')
        
        print(f"Output: {output:.2f}, Label: {label:.2f}, Error: {error:.2f}, Percent Error: {percent_error:.2f}%")
    
    # 이미지 셋 이름을 출력합니다.
    image_set_names = [img_set for img_set, _, _ in dataset.image_list]
    print("Image Set Names:", image_set_names)

if __name__ == "__main__":
    main()
