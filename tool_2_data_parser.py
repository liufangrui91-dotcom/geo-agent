import json
from openai import OpenAI
from typing import Dict, Any
from config import API_KEY,BASE_URL, MODEL_NAME

# 在正规Agent框架中，密钥应通过环境变量 os.environ.get() 动态加载，避免硬编码
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """
【角色设定】
    你是一个严谨的地质工程数据解析引擎。你的唯一任务是将 Markdown 格式的边坡现场调查表转化为结构化的 JSON 数据。

    【核心提取规则】
    你需要输出一个包含 "basic_info" 和 "score_data" 两个主键的 JSON 对象。

    ▶ 规则 1：解析 basic_info（基本信息）- 必须使用【强制对照字典】
    扫描文本的表头、表尾及备注。当你遇到以下中文信息时，**必须严格使用对应指定的英文键名**，绝对不允许自己发明新名字或随意翻译！
    
    【强制对照字典】（如文本中该项为 "/" 或空，请赋值 null）：
    - 路线名称 -> route_name (字符串)
    - 边坡编号 -> slope_code (如 "SS80K46R336"，这是极其重要的核心ID，字符串)
    - 路线编号 -> route_code
    - 边坡位置 -> slope_position
    - 起始桩号 -> start_chainage
    - 终点桩号 -> end_chainage
    - 边坡类型 -> slope_type (严格取 "土质", "岩质", "二元")
    - 挖填类型 -> cut_fill_type
    - 边坡高度（m） -> height_m (纯数字浮点数)
    - 分级坡高（m） -> graded_height_m（如文本中该项为 "/" 或空，请赋值 null）
    - 边坡坡度（°） -> slope_angle_deg (纯数字或字符串)
    - 边坡长度（m） -> length_m (纯数字浮点数)
    - 坡面防护形式 -> surface_protection
    - 坡面加固形式 -> surface_reinforcement
    - 地表排水设施 -> surface_drainage
    - 地下排水设施 -> underground_drainage
    - 有无检查踏步 -> has_inspection_step (填 "有"或"无")
    - 地震基本烈度 -> seismic_intensity

    ▶ 规则 2：解析 score_data（打分数据）
    必须提取 Markdown 表格中的所有评分细则，按行构建对象列表。
    每个对象必须严格包含以下 4 个键：
    - item (项目编号，如 "A1", "F1"，字符串)
    - weight (规定的权重或分值，如 100。若原文该项为空或无权重，必须输出 null)
    - score (实际调查得分，数字)
    - result (现场情况描述，必须一字不差地摘录原文，字符串)
    // 必须提取表格中所有的 A-V 指标

    【输出格式约束】
    1. 必须输出合法的纯 JSON 格式。
    2. 绝对不要在开头和结尾添加 ```json 和 ``` 标记，不要包含任何解释性文本。

    【期望输出示例骨架】
    {
      "basic_info": {
        "route_name": "某某高速",
        "slope_id": "K046+272-K046+400右幅",
        "slope_type": "二元",
        "height_m": 41.3,
        "length_m": 138.0,
        "investigator": "贺邵",
        "date": "2023-10-15",
        "protection_type": "挡墙"
      },
      "score_data": [
        {
        // 必须提取表格中所有的 A-V 指标
          "item": "A1",
          "weight": 100,
          "score": 40,
          "result": "边坡上部地表水体..."
        }
      ]
    }
    【注意】
    1. 返回纯 JSON。
    2. 权重和得分必须转为数字类型（float/int）。
    3. 严格对应 Markdown 表格的行，不要遗漏。
"""

def parse_slope_text_to_dict(raw_content: str) -> Dict[str, Any]:
    """
    地质现场调查数据结构化解析工具。
    当需要从非结构化的文本或OCR识别结果中提取边坡特征、尺寸参数和病害评分时，调用此工具。
    
    参数:
        raw_content (str): OCR识别后输出的文本，或自然语言描述的边坡调查内容。
        
    返回:
        Dict: 包含 'basic_info' 和 'score_data' 的结构化字典。若解析失败，返回含 'error' 键的字典。
    """
    if not raw_content or not isinstance(raw_content, str):
        return {"error": "输入内容为空或非字符串格式"}

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result_str = response.choices[0].message.content
        parsed_data = json.loads(result_str)
        return parsed_data
        
    except json.JSONDecodeError:
        return {"error": "大模型返回的数据无法解析为标准JSON字典"}
    except Exception as e:
        return {"error": f"API调用发生内部异常: {str(e)}"}