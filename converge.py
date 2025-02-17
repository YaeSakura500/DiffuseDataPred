import os
import torch

def save_model(model, filename):
    # 保存模型
    save_path = os.path.join('./model', f"{model.__class__.__name__}_best.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_and_save_models(model_dir='./models'):
    for filename in os.listdir(model_dir):
        if filename.endswith('.pt'):
            file_path = os.path.join(model_dir, filename)
            try:
                # 尝试加载模型
                model = torch.load(file_path)
                
                if isinstance(model, torch.nn.DataParallel):
                    print(f"{filename} is a DataParallel model.")
                    # 将 DataParallel 中的模型提取出来
                    model = model.module
                
                print(f"{filename} is a {model.__class__.__name__} model.")
                save_model(model, filename)
            
            except Exception as e:
                print(f"Error loading {filename}: {e}")

# 执行函数
load_and_save_models()
