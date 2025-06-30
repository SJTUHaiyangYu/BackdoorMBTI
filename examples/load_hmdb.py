from torchvision import transforms
from torchvision.datasets import HMDB51
from torch.utils.data import DataLoader

root_path = "../data/hmdb51/data"
annotation_path = "../data/hmdb51/test_train_splits"
dataset = HMDB51(
    root=root_path,
    annotation_path=annotation_path,
    frames_per_clip=16,
    train=False,
    transform=None,
    num_workers=4,
)
print(dataset[0])
# import cv2
# import os

# def check_video(file_path):
#     cap = cv2.VideoCapture(file_path)
#     if not cap.isOpened():
#         print(f"{file_path} is invalid.")
#         return False
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#     cap.release()
#     # print(f"{file_path} is valid.")
#     return True

# def main():
#     video_dir = "../data/hmdb51/data"
#     for root, dirs, files in os.walk(video_dir):
#         for file in files:
#             if file.endswith(".avi"):
#                 file_path = os.path.join(root, file)
#                 check_video(file_path)

# if __name__ == "__main__":
#     main()
