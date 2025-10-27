#!/usr/bin/env python3
"""
手动下载WN18RRInductive数据集文件的脚本
解决SSL连接问题
"""

import os
import ssl
import urllib.request
import requests
from urllib.error import URLError

def download_file_with_retry(url, filepath, max_retries=3):
    """使用多种方法下载文件"""
    
    print(f"📥 开始下载: {url}")
    print(f"📁 保存到: {filepath}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 方法1: 使用urllib.request with SSL context
    for attempt in range(max_retries):
        try:
            print(f"🔄 尝试方法1 (urllib.request) - 第 {attempt + 1} 次")
            
            # 创建SSL上下文
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 创建请求
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            # 下载文件
            with urllib.request.urlopen(request, context=ssl_context, timeout=30) as response:
                with open(filepath, 'wb') as f:
                    f.write(response.read())
            
            print(f"✅ 方法1成功下载: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 方法1失败: {e}")
            if attempt < max_retries - 1:
                print("⏳ 等待2秒后重试...")
                import time
                time.sleep(2)
    
    # 方法2: 使用requests库
    try:
        print("🔄 尝试方法2 (requests库)")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ 方法2成功下载: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ 方法2失败: {e}")
    
    # 方法3: 使用wget命令
    try:
        print("🔄 尝试方法3 (wget命令)")
        import subprocess
        
        cmd = ['wget', '--no-check-certificate', '-O', filepath, url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✅ 方法3成功下载: {filepath}")
            return True
        else:
            print(f"❌ 方法3失败: {result.stderr}")
            
    except Exception as e:
        print(f"❌ 方法3失败: {e}")
    
    print(f"❌ 所有下载方法都失败了: {url}")
    return False

def download_wn18rr_inductive():
    """下载WN18RRInductive数据集"""
    
    print("=" * 60)
    print("🌐 手动下载WN18RRInductive数据集")
    print("=" * 60)
    
    # 数据集URLs
    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_v1_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_v1_ind/valid.txt", 
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_v1_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_v1/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_v1/valid.txt"
    ]
    
    # 文件路径
    base_dir = "/T20030104/ynj/semma/kg-datasets/grail/IndWN18RR/v1/raw"
    filenames = [
        "train_ind.txt", "valid_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
    ]
    
    success_count = 0
    total_count = len(urls)
    
    for url, filename in zip(urls, filenames):
        filepath = os.path.join(base_dir, filename)
        
        if download_file_with_retry(url, filepath):
            success_count += 1
        
        print("-" * 40)
    
    print(f"\n📊 下载结果:")
    print(f"   - 成功: {success_count}/{total_count}")
    print(f"   - 失败: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 所有文件下载成功!")
        print("现在可以运行您的训练命令了:")
        print("python script/run.py -c config/inductive/inference.yaml --dataset WN18RRInductive --version v1 --ckpt /T20030104/ynj/semma/ckpts/semma.pth --gpus [0] --bpe null --epochs 0")
    else:
        print("❌ 部分文件下载失败，请检查网络连接或手动下载")
    
    print("=" * 60)

if __name__ == "__main__":
    download_wn18rr_inductive()
