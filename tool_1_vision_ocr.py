import os
import base64
from typing import Dict, Any
from openai import OpenAI
from config import API_KEY,BASE_URL, MODEL_NAME


# =========================================================================
# 1. 基础配置
# =========================================================================

def encode_image(image_path: str) -> str:
    """将本地图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# =========================================================================
# 2. 智能体工具执行核心
# =========================================================================
def extract_text_from_image(input_path: str) -> Dict[str, Any]:
    """视觉 OCR 提取工具（支持多图连扫）"""
    if not input_path or not os.path.exists(input_path):
        return {"error": f"找不到指定的路径: {input_path}"}
        
    # 1. 收集需要扫描的图片路径（过滤掉全景照）
    image_paths = []
    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            # 核心规则：跳过 image_ 开头的文件
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and not filename.lower().startswith("image_"):
                image_paths.append(os.path.join(input_path, filename))
    elif os.path.isfile(input_path):
        image_paths.append(input_path)
        
    if not image_paths:
        return {"error": "未找到需要 OCR 的表格图片（注：image_开头的图片将被视为全景图跳过OCR识别）"}

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    all_ocr_texts = []
    
    # 2. 逐张扫描并拼接结果
    for img_path in sorted(image_paths):
        print(f"   📷 [Tool 1] 正在识别图像特征: {os.path.basename(img_path)}")
        try:
            base64_image = encode_image(img_path)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "你是一个专业的地质工程资料录入员。请提取这张图片中的所有表格信息和文字，转换为 Markdown 格式的表格输出。不要省略任何信息。"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                temperature=0.1
            )
            all_ocr_texts.append(response.choices[0].message.content)
        except Exception as e:
            print(f"   ⚠️ [Tool 1] 图片 {os.path.basename(img_path)} 识别失败: {str(e)}")
            
    return {
        "status": "success", 
        "ocr_text": "\n\n".join(all_ocr_texts)  # 将多张表格的解析结果拼接成完整长文本
    }