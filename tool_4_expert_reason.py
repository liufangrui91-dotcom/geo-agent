import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from openai import OpenAI
from config import API_KEY,BASE_URL, MODEL_NAME

# =========================================================================
# 1. 规则与输出结构定义
# =========================================================================
SLOPE_CLASSIFICATION_RULES = {
    "土质边坡": {"逻辑": "I类(TS<40); II类(50>TS≥40); III类(60>TS≥50); IV类(TS≥60)"},
    "岩质边坡": {"逻辑": "I类(TS<50); II类(65>TS≥50); III类(80>TS≥65); IV类(TS≥80)"},
    "二元边坡": {"逻辑": "I类(TS<40); II类(55>TS≥40); III类(70>TS≥55); IV类(TS≥70)"}
}

class StabilityAssessment(BaseModel):
    slope_id: str = Field(description="边坡编号")
    thinking_process: str = Field(description="思维链推理过程：梳理病害与TS分数、定级的逻辑关系")
    slope_type: str = Field(description="边坡类型")
    stability_level: str = Field(description="风险等级 (I类/II类/III类/IV类)")
    key_factors: list[str] = Field(description="主要影响因素，语言极其精炼")
    reasoning: str = Field(description="符合核心期刊标准的学术推理段落")
    suggested_action: str = Field(description="防控对策")

# =========================================================================
# 2. 智能体工具执行核心 (依赖注入模式)
# =========================================================================
def generate_expert_reasoning(
    parsed_data: Dict[str, Any], 
    calc_result: Dict[str, Any], 
    retriever: Optional[Any] = None
) -> Dict[str, Any]:
    """
    RAG 专家软推理工具。
    接收解析后数据与硬计算结果，结合外部注入的规范检索器，生成风险评估段落。
    
    参数:
        parsed_data (Dict): 由结构化解析工具提取出的边坡参数字典。
        calc_result (Dict): 由物理规则硬计算工具算出的结果字典 (包含 TS, Level)。
        retriever (Any): 外部传入的 LangChain Chroma 检索器对象。
        
    返回:
        Dict: 包含学术推理段落的字典。
    """
    if not parsed_data or not calc_result:
        return {"error": "缺少解析数据或硬计算结果，阻断推理"}

    basic_info = parsed_data.get("basic_info", {})
    score_data = parsed_data.get("score_data", [])
    calc_data = calc_result.get("calculation_result", {})
    
    # 数据清洗与映射
    slope_id = basic_info.get("slope_code") or basic_info.get("slope_id", "Unknown")
    raw_type = basic_info.get("slope_type", "二元")
    slope_type = raw_type if "边坡" in raw_type else raw_type + "边坡"
    ts_value = calc_data.get("TS", 0.0)

    # 提取病害特征
    ab_issues = []
    for item in score_data:
        item_code = str(item.get("item", "")).upper()
        if item_code.startswith("A") or item_code.startswith("B"):
            try:
                if float(item.get("score", 0)) > 0:
                    res_text = str(item.get("result", "")).strip()
                    if res_text and res_text not in ["/", "无", "无破损", "结构完好", "None", "null"]:
                        ab_issues.append(f"[{item_code}] {res_text}")
            except ValueError:
                pass
    disease_features = "；".join(ab_issues) if ab_issues else "无明显扣分病害"

    # RAG 动态检索
    context_text = "未检索到额外条文，请依据 System Prompt 逻辑推理"
    if retriever:
        try:
            query = f"{slope_type}高风险评估，TS评分为{ts_value}，实测特征：{disease_features}"
            docs = retriever.invoke(query)
            context_text = "\n".join([d.page_content for d in docs])
        except Exception as e:
            print(f"RAG 检索器抛出异常: {e}")

    # 此处为保障独立运行，从环境变量获取密钥。正式编排时可改由 config 统一管控
    api_key = os.environ.get("SILICON_API_KEY", "sk-rlyzoxflrjhkpuuoypgsvuhjvzulludwvmytnqxcgetcfxrr")
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=0.1,
        max_tokens=1500,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    parser = JsonOutputParser(pydantic_object=StabilityAssessment)
    rules = SLOPE_CLASSIFICATION_RULES.get(slope_type, SLOPE_CLASSIFICATION_RULES["二元边坡"])

    # === 替换从 system_template 开始，直到 chain = prompt | llm | parser 的代码 ===

    # 注意：这里去掉了最外层的 f，变成普通的字符串模板，使用 {} 留出插槽
    system_template = """
    你是我撰写中文高质量学术期刊的专属“Geo-Agent科研搭档”，同时也是湖南省顶尖的地质勘察工程师。你的研判必须严格依据《湖南省高风险边坡评估技术指南（修订稿）》。
    
    【评估对象】
    类型：{slope_type}
    定量基准：{rules_logic}
    
    【⚠️ 学术推理与文风强制约束】
    1. 坚守 TS 评价体系：输入数据中的 "TS" 值为硬计算算出的综合评分。严禁提及“安全系数(Fs)”。
    2. 风险等级认知：“I类”代表风险极低，数字越大代表风险越高。
    3. 辩证评估原则：严禁无脑使用“未见明显病害”！必须依据输入的“现场实测病害特征”客观描述。
    4. 工程勘察文风：彻底消除AI生成的机械痕迹。起手式必须采用：“按照《湖南省高风险边坡评估技术指南（修订稿）》相关评价标准，该边坡定量计算总评分TS为[X]分。”
    
    【🧠 强制思维链】必须在 JSON 中先输出 thinking_process 字段。
    """

    # 同样去掉 f，作为模板字符串
    human_template = """
    【RAG 规范上下文】: {context_text}
    
    【输入多源异构数据】:
    - 评估单元编号: {slope_id}
    - 边坡地质类型: {slope_type}
    - 定量计算总评分(TS): {ts_value}
    - 现场实测病害特征: {disease_features}
    
    【任务】: 依据上述条件，执行评估并输出 JSON。
    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", human_template)
    ])

    chain = prompt | llm | parser

    try:
        # 在 invoke 时，以字典的形式把所有的变量（包括带有复杂大括号的指令）传进去
        result = chain.invoke({
            "slope_type": slope_type,
            "rules_logic": rules['逻辑'],
            "context_text": context_text,
            "slope_id": slope_id,
            "ts_value": ts_value,
            "disease_features": disease_features,
            "format_instructions": parser.get_format_instructions()
        })
        return {
            "status": "success",
            "reasoning_result": result
        }
    except Exception as e:
        return {"error": f"LLM推理生成失败: {str(e)}"}

  