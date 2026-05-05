import re
from typing import Dict, Any

# =========================================================================
# 1. 基础数据与常量定义 (从原 03_hard_calc.py 迁移)
# =========================================================================
WEIGHTS = {
    'soil': {'A1':0.056,'A2':0.041,'A3':0.077,'A4':0.032,'A5':0.163,'B1':0.149,'B2':0.046,'B3':0.185,'B4':0.149,'C1':0.029,'C2':0.043,'C3':0.029},
    'binary': {'A1':0.032,'A2':0.023,'A3':0.056,'A4':0.079,'B1':0.081,'B2':0.242,'B3':0.123,'C1':0.021,'C2':0.063,'C3':0.041,'C4':0.055,'C5':0.071,'C6':0.018,'C7':0.094},
    'rock_collapse': {'A1':0.018,'A2':0.009,'A3':0.003,'A4':0.025,'B1':0.092,'B2':0.367,'B3':0.230,'C1':0.030,'C2':0.046,'C10':0.049,'C11':0.118,'C12':0.012},
    'rock_toppling': {'A1':0.020,'A2':0.011,'A3':0.007,'A4':0.018,'B1':0.084,'B2':0.336,'B3':0.236,'C1':0.029,'C2':0.086,'C3':0.011,'C4':0.020,'C5':0.018,'C6':0.028,'C7':0.054,'C10':0.044},
    'rock_planar': {'A1':0.019,'A2':0.010,'A3':0.007,'A4':0.018,'B1':0.084,'B2':0.334,'B3':0.229,'C1':0.045,'C2':0.061,'C3':0.011,'C4':0.013,'C5':0.017,'C6':0.016,'C8':0.089,'C10':0.048},
    'rock_wedge': {'A1':0.020,'A2':0.011,'A3':0.007,'A4':0.018,'B1':0.084,'B2':0.336,'B3':0.236,'C1':0.050,'C2':0.065,'C3':0.014,'C4':0.016,'C5':0.024,'C6':0.031,'C9':0.046,'C10':0.044}
}

LOOKUP_TABLE = {
    0.3: {0.0:(0.5,0.5), 0.3:(0.5,0.6), 0.6:(0.5,0.7), 0.9:(0.5,0.8), 1.2:(0.5,0.9), 1.5:(0.5,1.0)},
    0.5: {0.0:(0.8,1.0), 0.3:(0.8,1.2), 0.6:(0.8,1.4), 0.9:(0.8,1.6), 1.2:(0.8,1.8), 1.5:(0.8,2.0)},
    0.7: {0.0:(1.0,1.3), 0.3:(1.0,1.5), 0.6:(1.0,1.7), 0.9:(1.0,2.0), 1.2:(1.0,2.3), 1.5:(1.0,2.6)},
    1.0: {0.0:(1.2,1.5), 0.3:(1.2,1.8), 0.6:(1.2,2.1), 0.9:(1.2,2.4), 1.2:(1.2,2.7), 1.5:(1.2,3.0)}
}

def safe_float(val: Any, default=0.0) -> float:
    if val is None: return default
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    if match: return float(match.group())
    return default

# =========================================================================
# 2. 智能体工具执行核心
# =========================================================================
def calculate_slope_ts(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    边坡物理规则硬计算工具。
    当已经获得了结构化的边坡特征参数字典（包含基本信息和各扣分项打分），且需要计算稳定系数（TS）和风险等级（Level）时，调用此工具。
    
    参数:
        parsed_data (Dict): 由结构化解析工具提取出的边坡参数字典。
        
    返回:
        Dict: 包含 IS, CS, TS 评分以及最终风险 Level 的计算结果字典。
    """
    if not parsed_data or "basic_info" not in parsed_data:
        return {"error": "输入的字典数据缺失基础结构（basic_info）"}
        
    try:
        basic = parsed_data.get("basic_info", {})
        scores = parsed_data.get("score_data", [])
        
        # 1. 提取边坡类型与高度
        raw_type = basic.get("slope_type", "soil")
        stype = "rock" if "岩" in raw_type else "binary" if "二元" in raw_type else "soil"
        
        H_real = 0.0
        for item in scores:
            if str(item.get("item", "")).strip().upper() == "C1":
                H_real = safe_float(item.get("result", 0.0))
                break
                
        H = H_real if H_real > 0 else safe_float(basic.get("height_m"), 0.0)
        score_dict = {str(item.get("item", "")).strip().upper(): safe_float(item.get("score"), 0.0) for item in scores if item.get("item")}
        
        # 2. 映射 CS 计算参数
        if stype == "binary":
            size_val, terr_val = score_dict.get("M", 0.5), score_dict.get("K", 0.0)
            top_s, top_d = score_dict.get("G1", 0.1), score_dict.get("G2", 0.0)
            toe_s, toe_d = score_dict.get("J1", 1.0), score_dict.get("J2", 0.0)
        else:
            size_val, terr_val = score_dict.get("K", 0.5), score_dict.get("J", 0.0)
            top_s, top_d = score_dict.get("F1", 0.1), score_dict.get("F2", 0.0)
            toe_s, toe_d = score_dict.get("G1", 1.0), score_dict.get("G2", 0.0)
            
        V = score_dict.get("V", 1.5)

        # 3. 执行核心计算逻辑
        safe_H = max(1.0, min(H, 30.0))

        avail_sizes = sorted(list(LOOKUP_TABLE.keys()))
        closest_size = min(avail_sizes, key=lambda x: abs(x - size_val))
        avail_terrains = sorted(list(LOOKUP_TABLE[closest_size].keys()))
        closest_terrain = min(avail_terrains, key=lambda x: abs(x - terr_val))
        alpha, beta = LOOKUP_TABLE[closest_size][closest_terrain]

        denom1, denom2 = alpha * safe_H, beta * safe_H
        term1 = max(0, top_s * (alpha * safe_H - top_d) / denom1 if denom1 != 0 else 0.0)
        term2 = max(0, 2 * toe_s * (beta * safe_H - toe_d) / denom2 if denom2 != 0 else 0.0)

        CS = size_val * (term1 + term2) * safe_H * V
        IS = 0.0
        
        if stype == 'soil': 
            IS = sum(safe_float(score_dict.get(k)) * w for k, w in WEIGHTS['soil'].items())
        elif stype == 'binary': 
            IS = sum(safe_float(score_dict.get(k)) * w for k, w in WEIGHTS['binary'].items())
        elif stype == 'rock':
            vals = [sum(safe_float(score_dict.get(k)) * w for k, w in WEIGHTS[m].items()) for m in ['rock_collapse', 'rock_toppling', 'rock_planar', 'rock_wedge']]
            IS = max(vals) if vals else 0

        TS = IS * CS / 100.0
        
        # 4. 判定风险等级
        level = "I类"
        if stype == 'soil': level = "IV类" if TS >= 60 else "III类" if TS >= 50 else "II类" if TS >= 40 else "I类"
        elif stype == 'rock': level = "IV类" if TS >= 80 else "III类" if TS >= 65 else "II类" if TS >= 50 else "I类"
        elif stype == 'binary': level = "IV类" if TS >= 70 else "III类" if TS >= 55 else "II类" if TS >= 40 else "I类"

        return {
            "status": "success",
            "calculation_result": {
                "IS": round(IS, 2), 
                "CS": round(CS, 2), 
                "TS": round(TS, 2), 
                "Level": level
            }
        }
        
    except Exception as e:
        return {"error": f"硬计算过程发生数学运算或数据类型异常: {str(e)}"}