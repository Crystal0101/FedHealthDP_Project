import yaml
from scripts.preprocess import preprocess_data
from scripts.train import train_model
from scripts.evaluate import evaluate_model

def main():
    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 数据预处理
    data = preprocess_data(config['data'])
    
    # 模型训练
    model = train_model(data, config['training'])
    
    # 模型评估
    evaluate_model(model, data['test'])
    
if __name__ == "__main__":
    main()
