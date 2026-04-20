"""
柑橘检测 Skill 模块
封装单视角 ConvNeXt-Tiny 模型，提供图像分类推理功能
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import os

# 添加项目根目录到路径
CHONGJU_ROOT = r"C:\Users\18656\Desktop\daima\chongju"
sys.path.insert(0, CHONGJU_ROOT)

from models.comparison_models import SingleViewModel

# 创建简单的配置类（避免导入冲突）
class SimpleConfig:
    BACKBONE_MODEL = 'convnext_tiny'
    NUM_CLASSES = 2
    DROPOUT_RATE = 0.3
    FREEZE_CONVNEXT = False

ProjectConfig = SimpleConfig()


class CitrusDetectionSkill:
    """
    柑橘检测 Skill
    加载训练好的 SingleViewModel，提供推理接口
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化检测器
        
        Args:
            model_path: 模型权重文件路径
            device: 运行设备 (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.class_names = ["健康", "患虫"]
        self.image_size = 224
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载模型
        self.model = self._load_model()
        print(f"✓ 柑橘检测 Skill 初始化完成，设备：{self.device}")
    
    def _load_model(self):
        """加载训练好的模型"""
        # 创建模型实例
        model = SingleViewModel(ProjectConfig, backbone_model='convnext_tiny')
        
        # 加载权重
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 处理不同的 checkpoint 格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    # 移除可能的前缀
                    state_dict = {k.replace('module.', ''): v 
                                 for k, v in checkpoint['state_dict'].items()}
                    model.load_state_dict(state_dict)
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            print(f"✓ 成功加载模型权重：{self.model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def predict(self, image_path: str = None, image_pil: Image.Image = None):
        """
        预测单张图像
        
        Args:
            image_path: 图像文件路径，或
            image_pil: PIL Image 对象
            
        Returns:
            dict: {
                'class': 类别名称，
                'class_id': 类别 ID,
                'confidence': 置信度，
                'probabilities': [健康概率，患虫概率]
            }
        """
        # 加载图像
        if image_path:
            image = Image.open(image_path).convert('RGB')
        elif image_pil:
            image = image_pil
        else:
            raise ValueError("必须提供 image_path 或 image_pil")
        
        # 预处理单张图像
        single_tensor = self.transform(image)  # (3, 224, 224)
        
        # 复制 5 份以匹配模型输入要求 (5, 3, 224, 224)
        # 因为 SingleViewModel 期望 5 视角输入，但只使用第一张
        input_tensor = single_tensor.unsqueeze(0).repeat(5, 1, 1, 1)  # (5, 3, 224, 224)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # (1, 5, 3, 224, 224)
        
        # 推理
        with torch.no_grad():
            logits, probabilities = self.model(input_tensor)
            probs = probabilities.cpu().numpy()[0]
            predicted_class = np.argmax(probs)
            confidence = float(probs[predicted_class])
        
        return {
            'class': self.class_names[predicted_class],
            'class_id': int(predicted_class),
            'confidence': round(confidence, 4),
            'probabilities': {
                '健康': round(float(probs[0]), 4),
                '患虫': round(float(probs[1]), 4)
            }
        }
    
    def predict_from_base64(self, base64_string: str):
        """
        从 base64 字符串预测
        
        Args:
            base64_string: base64 编码的图像字符串
            
        Returns:
            dict: 预测结果
        """
        import base64
        from io import BytesIO
        
        # 解码 base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        return self.predict(image_pil=image)


# 创建全局实例（延迟加载）
_detection_skill = None


def get_detection_skill():
    """获取检测 Skill 实例（单例模式）"""
    global _detection_skill
    if _detection_skill is None:
        from config import SINGLE_VIEW_MODEL_PATH
        _detection_skill = CitrusDetectionSkill(SINGLE_VIEW_MODEL_PATH)
    return _detection_skill


if __name__ == "__main__":
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        skill = get_detection_skill()
        result = skill.predict(sys.argv[1])
        print(f"预测结果：{result}")
    else:
        print("用法：python skills/detection_skill.py <图像路径>")
