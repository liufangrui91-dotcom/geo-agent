import os
from typing import Literal
from langgraph.graph import StateGraph, END
from agent_core.state_manager import AgentState

# 导入已完成重构的核心工具
from tools.tool_1_vision_ocr import extract_text_from_image
from tools.tool_2_data_parser import parse_slope_text_to_dict
from tools.tool_3_physics_calc import calculate_slope_ts
from tools.tool_4_expert_reason import generate_expert_reasoning
from tools.tool_5_report_writer import generate_word_report
from tools.tool_2b_disease_vision import generate_disease_description

# =========================================================================
# 1. 定义节点 (Nodes)
# =========================================================================

def node_tool_1_ocr(state: AgentState) -> dict:
    """工具1：视觉OCR提取"""
    print("[运行节点] Tool 1: 视觉 OCR 提取")
    image_path = state.get("raw_input_path", "")
    
    result = extract_text_from_image(image_path)
    
    # 将提取结果写入状态字典，传给下一个节点
    return {"ocr_text": result.get("ocr_text", "")}

def node_tool_2_parser(state: AgentState) -> dict:
    """工具2：数据结构化解析"""
    print("[运行节点] Tool 2: 数据结构化解析")
    # 优先读取 OCR 文本，如果没有则直接读取原始输入
    input_text = state.get("ocr_text") or state.get("raw_input_path", "")
    result = parse_slope_text_to_dict(input_text)
    return {"parsed_data": result}


def node_tool_2b_vision(state: AgentState) -> dict:
    """工具2b：图文交叉病害感知"""
    print("[运行节点] Tool 2b: 图文交叉病害感知")
    parsed_data = state.get("parsed_data", {})
    raw_input_path = state.get("raw_input_path", "")
    
    result = generate_disease_description(parsed_data, raw_input_path)
    return {"disease_description": result.get("disease_description", "")}

def node_tool_3_calc(state: AgentState) -> dict:
    """工具3：物理硬计算"""
    print("[运行节点] Tool 3: 物理规则硬计算")
    parsed_data = state.get("parsed_data", {})
    result = calculate_slope_ts(parsed_data)
    return {"calculation_results": result}


def node_tool_5_report(state: AgentState) -> dict:
    """工具5：Word 报告渲染"""
    print("[运行节点] Tool 5: 报告渲染输出")
    
    # 1. 基础数据抓取
    parsed_data = state.get("parsed_data", {})
    calc_results = state.get("calculation_results", {})
    reasoning_text = state.get("reasoning_text", "")
    raw_input_path = state.get("raw_input_path", "")
    
    # 2. 新增数据抓取 (病害描述与自定义路径)
    disease_desc = state.get("disease_description", "无病害描述。")
    out_dir = state.get("output_dir") 
    
    # 3. 注入渲染引擎
    result = generate_word_report(
        parsed_data=parsed_data,
        calc_result=calc_results,
        reasoning_text=reasoning_text,
        raw_input_path=raw_input_path,
        disease_description=disease_desc, 
        output_dir=out_dir                
    )
    
    return {"report_path": result.get("report_path", "")}

# =========================================================================
# 2. 定义路由逻辑 (Routing Logic)
# =========================================================================

def route_input(state: AgentState) -> Literal["node_tool_1", "node_tool_2"]:
    """判断入口流向。兼容单图片和文件夹目录。"""
    raw_input = state.get("raw_input_path", "")
    
    # 新增 os.path.isdir 判断
    if os.path.isdir(raw_input) or raw_input.lower().endswith(('.png', '.jpg', '.jpeg')):
        print("  -> [决策路由] 识别为图像目录或文件，导向 Tool 1 (OCR)")
        return "node_tool_1"
    else:
        print("  -> [决策路由] 识别为文本输入，导向 Tool 2 (解析)")
        return "node_tool_2"

# =========================================================================
# 3. 编译状态图 (Build Graph)
# =========================================================================

# ...前面 node_tool_1 到 node_tool_3 的代码保持不变 ...

def build_workflow(retriever=None) -> StateGraph:
    """构建并编译智能体工作流"""
    workflow = StateGraph(AgentState)

    # 闭包：在内部重新定义 tool_4 的调用逻辑，以便捕获外部传入的 retriever
    def node_tool_4_reason_with_retriever(state: AgentState) -> dict:
        print("[运行节点] Tool 4: RAG 专家软推理")
        parsed_data = state.get("parsed_data", {})
        calc_results = state.get("calculation_results", {})
        
        # 将接收到的真实检索器注入到核心工具中
        result = generate_expert_reasoning(parsed_data, calc_results, retriever=retriever)
        return {"reasoning_text": result.get("reasoning_result", {}).get("reasoning", "推理失败")}

    workflow.add_node("node_tool_1", node_tool_1_ocr)
    workflow.add_node("node_tool_2", node_tool_2_parser)
    workflow.add_node("node_tool_2b", node_tool_2b_vision)
    workflow.add_node("node_tool_3", node_tool_3_calc)
    workflow.add_node("node_tool_4", node_tool_4_reason_with_retriever) # 使用闭包节点
    workflow.add_node("node_tool_5", node_tool_5_report)

    workflow.set_conditional_entry_point(
        route_input,
        {
            "node_tool_1": "node_tool_1",
            "node_tool_2": "node_tool_2"
        }
    )

    workflow.add_edge("node_tool_1", "node_tool_2")
    workflow.add_edge("node_tool_2", "node_tool_2b")  # 修改此处
    workflow.add_edge("node_tool_2b", "node_tool_3")  # 修改此处
    workflow.add_edge("node_tool_3", "node_tool_4")
    workflow.add_edge("node_tool_4", "node_tool_5")
    workflow.add_edge("node_tool_5", END)

    return workflow.compile()