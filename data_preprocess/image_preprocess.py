"""
图像预处理模块
功能：实现图像读取、视频抽帧、数据增强、预处理等功能
作者：柑橘虫害检测项目组
日期：2026年
"""
import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List, Union, Optional
from PIL import Image, ImageEnhance


class ImagePreprocessor:
    def __init__(self, image_size: int = 224, is_train: bool = True):
        self.image_size = image_size
        self.is_train = is_train
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                self.normalize
            ])
        
    def read_image(self, image_path: str) -> Image.Image:
        """
        读取图像文件
        
        参数:
            image_path: 图像文件路径
        
        返回:
            PIL图像对象
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"无法读取图像 {image_path}: {str(e)}")
            
    def extract_frames_from_video(self, video_path: str, num_frames: int = 4) -> List[Image.Image]:
        """
        从水平环绕视频中按时间戳四等分提取帧
        
        参数:
            video_path: 视频文件路径
            num_frames: 需要提取的帧数（默认4张，对应0°,90°,180°,270°）
        
        返回:
            提取的帧图像列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件 {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
        
        if total_duration <= 0:
            raise ValueError(f"视频时长计算失败: {video_path}")
            
        time_points = np.linspace(0, total_duration, num_frames, endpoint=False)
        
        frames = []
        for t in time_points:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                
        cap.release()
        
        if len(frames) != num_frames:
            raise ValueError(f"期望提取 {num_frames} 帧，实际提取 {len(frames)} 帧")
            
        return frames
        
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        预处理单张图像
        
        参数:
            image: 图像文件路径或PIL图像对象
        
        返回:
            预处理后的图像张量，形状为 (3, 224, 224)
        """
        if isinstance(image, str):
            image = self.read_image(image)
            
        return self.transform(image)
        
    def preprocess_images(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        """
        预处理多张图像
        
        参数:
            images: 图像文件路径列表或PIL图像对象列表
        
        返回:
            预处理后的图像张量，形状为 (5, 3, 224, 224)
        """
        if len(images) != 5:
            raise ValueError(f"输入图像数量应为5张（4张水平+1张顶角），当前为 {len(images)} 张")
            
        processed_tensors = []
        for image in images:
            processed_tensors.append(self.preprocess_image(image))
            
        return torch.stack(processed_tensors, dim=0)
        
    def preprocess(self, 
                   horizontal_images: Optional[List[Union[str, Image.Image]]] = None,
                   top_image: Optional[Union[str, Image.Image]] = None,
                   video_path: Optional[str] = None) -> torch.Tensor:
        """
        完整的图像数据预处理流程
        
        参数:
            horizontal_images: 4张水平环绕图像（可选，与video_path二选一）
            top_image: 1张顶角图像
            video_path: 水平环绕视频路径（可选，与horizontal_images二选一）
        
        返回:
            预处理后的图像张量，形状为 (5, 3, 224, 224)
        """
        if top_image is None:
            raise ValueError("必须提供顶角图像")
            
        if horizontal_images is None and video_path is None:
            raise ValueError("必须提供水平环绕图像或视频")
            
        if horizontal_images is not None:
            if len(horizontal_images) != 4:
                raise ValueError(f"水平环绕图像数量应为4张，当前为 {len(horizontal_images)} 张")
            all_images = horizontal_images + [top_image]
        else:
            horizontal_frames = self.extract_frames_from_video(video_path, 4)
            all_images = horizontal_frames + [top_image if isinstance(top_image, Image.Image) else self.read_image(top_image)]
            
        return self.preprocess_images(all_images)
