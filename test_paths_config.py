#!/usr/bin/env python3
"""
测试脚本：验证所有路径配置是否正确从 flags.yaml 加载
"""

import os
import sys
import yaml

def test_paths_config():
    """测试路径配置"""
    print("=" * 60)
    print("🔍 测试 flags.yaml 路径配置")
    print("=" * 60)
    
    # 加载配置
    mydir = os.getcwd()
    flags_path = os.path.join(mydir, "flags.yaml")
    
    if not os.path.exists(flags_path):
        print(f"❌ 错误: flags.yaml 文件不存在于 {flags_path}")
        return False
    
    print(f"✅ 找到 flags.yaml: {flags_path}")
    
    try:
        with open(flags_path, 'r', encoding='utf-8') as f:
            flags_data = yaml.safe_load(f)
        print("✅ 成功加载 flags.yaml")
        
        # 创建一个简单的 flags 对象
        class Flags:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        flags = Flags(flags_data)
    except Exception as e:
        print(f"❌ 加载 flags.yaml 失败: {e}")
        return False
    
    # 检查必需的路径配置
    required_paths = [
        'base_path',
        'ckpt_path',
        'models_path',
        'kg_datasets_path',
        'openrouter_path'
    ]
    
    print("\n📋 检查路径配置项:")
    all_exist = True
    for path_key in required_paths:
        if hasattr(flags, path_key):
            path_value = getattr(flags, path_key)
            exists = os.path.exists(path_value) if path_value else False
            status = "✅" if exists else "⚠️ (目录不存在，但配置存在)"
            print(f"   {status} {path_key}: {path_value}")
        else:
            print(f"   ❌ {path_key}: 未配置")
            all_exist = False
    
    if not all_exist:
        print("\n❌ 部分路径配置缺失!")
        return False
    
    # 测试路径使用场景
    print("\n🧪 测试路径使用场景:")
    
    # 1. 测试 openrouter 路径
    test_dataset = "FB15k237"
    json_path = os.path.join(flags.openrouter_path, "relations_type/gpt-4o-2024-11-20", f"{test_dataset}.json")
    print(f"   📁 OpenRouter JSON 路径示例: {json_path}")
    print(f"      存在: {'✅' if os.path.exists(json_path) else '⚠️ (文件不存在，但路径正确)'}")
    
    # 2. 测试模型路径
    model_path_sentbert = os.path.join(flags.models_path, "all-mpnet-base-v2")
    model_path_jina = os.path.join(flags.models_path, "jina-embeddings-v3")
    print(f"   🤖 SentBERT 模型路径: {model_path_sentbert}")
    print(f"      存在: {'✅' if os.path.exists(model_path_sentbert) else '⚠️ (目录不存在)'}")
    print(f"   🤖 JinaAI 模型路径: {model_path_jina}")
    print(f"      存在: {'✅' if os.path.exists(model_path_jina) else '⚠️ (目录不存在)'}")
    
    # 3. 测试 checkpoint 路径
    ckpt_path = os.path.join(flags.ckpt_path, "semma.pth")
    print(f"   💾 Checkpoint 路径示例: {ckpt_path}")
    print(f"      存在: {'✅' if os.path.exists(ckpt_path) else '⚠️ (文件不存在，但路径正确)'}")
    
    # 4. 测试数据集路径
    dataset_path = os.path.join(flags.kg_datasets_path, "grail/IndWN18RR/v1/raw")
    print(f"   📊 数据集路径示例: {dataset_path}")
    print(f"      存在: {'✅' if os.path.exists(dataset_path) else '⚠️ (目录不存在，但路径正确)'}")
    
    # 验证代码中使用的路径
    print("\n🔎 验证代码中的路径使用:")
    
    # 检查 ultra/tasks.py
    tasks_py_path = os.path.join(mydir, "ultra", "tasks.py")
    if os.path.exists(tasks_py_path):
        with open(tasks_py_path, 'r', encoding='utf-8') as f:
            tasks_content = f.read()
            if 'flags.openrouter_path' in tasks_content:
                print("   ✅ ultra/tasks.py 使用 flags.openrouter_path")
            else:
                print("   ❌ ultra/tasks.py 未使用 flags.openrouter_path")
                all_exist = False
            
            if 'flags.models_path' in tasks_content:
                print("   ✅ ultra/tasks.py 使用 flags.models_path")
            else:
                print("   ❌ ultra/tasks.py 未使用 flags.models_path")
                all_exist = False
            
            # 检查是否还有硬编码路径
            if '/T20030104/ynj/semma' in tasks_content:
                print("   ⚠️  ultra/tasks.py 中仍有硬编码路径 /T20030104/ynj/semma")
                all_exist = False
            else:
                print("   ✅ ultra/tasks.py 中没有硬编码路径")
    
    # 检查 download_dataset.py
    download_py_path = os.path.join(mydir, "download_dataset.py")
    if os.path.exists(download_py_path):
        with open(download_py_path, 'r', encoding='utf-8') as f:
            download_content = f.read()
            if 'flags.kg_datasets_path' in download_content:
                print("   ✅ download_dataset.py 使用 flags.kg_datasets_path")
            else:
                print("   ❌ download_dataset.py 未使用 flags.kg_datasets_path")
                all_exist = False
            
            if 'flags.ckpt_path' in download_content:
                print("   ✅ download_dataset.py 使用 flags.ckpt_path")
            else:
                print("   ❌ download_dataset.py 未使用 flags.ckpt_path")
                all_exist = False
            
            # 检查是否还有硬编码路径
            if '/T20030104/ynj/semma' in download_content:
                print("   ⚠️  download_dataset.py 中仍有硬编码路径 /T20030104/ynj/semma")
                all_exist = False
            else:
                print("   ✅ download_dataset.py 中没有硬编码路径")
    
    print("\n" + "=" * 60)
    if all_exist:
        print("✅ 所有路径配置测试通过!")
        return True
    else:
        print("❌ 部分测试未通过，请检查上述问题")
        return False

if __name__ == "__main__":
    success = test_paths_config()
    sys.exit(0 if success else 1)
