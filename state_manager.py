from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """智能体内存状态定义"""
    slope_id: str
    raw_input_path: str
    output_dir: Optional[str] # 新增输出目录字段
    ocr_text: Optional[str]
    parsed_data: Optional[Dict[str, Any]]
    
    # === 新增：用于存储 Tool 2b 生成的病害交叉验证描述 ===
    disease_description: Optional[str] 
    
    calculation_results: Optional[Dict[str, Any]]
    reasoning_text: Optional[str]
    report_path: Optional[str]
    errors: List[str]