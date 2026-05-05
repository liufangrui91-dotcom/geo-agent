import os
import base64
from typing import Dict, Any
from openai import OpenAI
from config import API_KEY, BASE_URL
from io import BytesIO
from PIL import Image

# 注意：视觉交叉验证需要支持多模态（Vision）的大模型。
# 请确保你在 SiliconFlow 中选用的是具备视觉能力的模型，例如 Qwen/Qwen-VL-Max 或 Pro/OpenAI/gpt-4o
VISION_MODEL_NAME = "Pro/moonshotai/Kimi-K2.6" # 此处以 InternVL2 为例，你可根据实际调用的视觉模型更改

def encode_image(image_path: str) -> str:
    """使用 PIL 清洗工程现场图片，过滤 MPO 等非标文件头，强转为 RGB JPEG"""
    with Image.open(image_path) as img:
        # 剥离可能存在的 Alpha 通道或 3D 深度图通道
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffered = BytesIO()
        # 强制以纯净的 JPEG 格式写入内存缓冲区
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_disease_description(parsed_data: Dict[str, Any], input_path: str) -> Dict[str, str]:
    """工具2b：基于扣分项与现场全景图的交叉验证病害描述"""
    
    # 1. 提取结构化打分表中的实测扣分项
    score_data = parsed_data.get("score_data", [])
    deduction_items = []
    for item in score_data:
        try:
            if float(item.get("score", 0)) > 0:
                res_text = str(item.get("result", "")).strip()
                if res_text and res_text not in ["/", "无", "无破损", "结构完好", "None", "null"]:
                    deduction_items.append(f"[{item.get('item')}] 现场记录：{res_text}")
        except ValueError:
            continue
            
    deduction_context = "；\n".join(deduction_items) if deduction_items else "调查表显示无明显扣分病害。"

    # 2. 寻找全景照片 (image_ 开头)
    image_paths = []
    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename.lower().startswith("image_"):
                image_paths.append(os.path.join(input_path, filename))
                
    if not image_paths:
        return {"disease_description": f"未提供现场全景图。依据调查表记录，边坡病害情况如下：\n{deduction_context}"}

    # 3. 组装多模态 Prompt (目前仅取第一张全景图进行描述)
    target_image = image_paths[0]
    base64_image = encode_image(target_image)
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    prompt_text = f"""
    你是一位严谨的地质工程勘察专家。请结合我提供的【现场实测调查表扣分项记录】以及【边坡现场全景图】，撰写一段约 200 字的“边坡主要病害描述”段落，用于直接插入勘察报告。注意，如果结合评分，发现边坡风险等级较低，也可直接描述为“边坡状况良好，未见明显病害特征”。
    【实测扣分项记录】：
    {deduction_context}
    
    【撰写要求】：
    1. 交叉验证：将文字记录中的病害与图片中展现的地貌特征进行对应描述。
    2. 客观专业：采用地质勘察专用术语，禁止使用抒情、夸张的修辞手法。
    3. 如果图片因被植被覆盖导致无法看清表记录的病害，请客观陈述“受坡面植被遮挡，表征不明显，依调查表记录为准”。
    4. 直接输出段落文本，不要任何寒暄、标题或多余的 Markdown 标记。
    """

    try:
        print(f"   👁️ [Tool 2b] 启动视觉感知，分析图片: {os.path.basename(target_image)}")
        response = client.chat.completions.create(
            model=VISION_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.2
        )
        description = response.choices[0].message.content.strip()
        return {"disease_description": description}
    except Exception as e:
        # 放入 except Exception as e: 代码块中
        print(f"   ⚠️ [Tool 2b] 视觉模型调用失败: {str(e)}。已触发文本 LLM 降级润色机制。")

        fallback_prompt = f"""
        你是一位严谨的地质工程勘察专家。
        请将以下【现场实测调查表扣分项记录】改写为一段连贯的“边坡主要病害描述”段落，用于勘察报告。
        要求：禁止列表式输出；使用专业地质术语将各扣分项串联成文。
        【记录】：
        {deduction_context}
        """

        try:
            # 这里的 model 请替换为你实际配置的纯文本 Qwen 模型名称
            fallback_response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V4-Flash", 
                messages=[{"role": "user", "content": fallback_prompt}],
                temperature=0.1
            )
            description = fallback_response.choices[0].message.content.strip()
            return {"disease_description": f"（注：无清晰全景图，依据调查表生成）{description}"}
        except Exception as fallback_e:
            # 最底层的防线，如果文本模型也挂了，再返回原始列表
            return {"disease_description": f"大模型解析全线异常。原始记录：{deduction_context}"}