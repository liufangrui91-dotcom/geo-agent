import os
import re
import io
from typing import Dict, Any
from PIL import Image, ImageFile

# 强行加载损坏/截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm

# =========================================================================
# 智能体工具执行核心: 报告渲染引擎
# =========================================================================
def generate_word_report(
    parsed_data: Dict[str, Any],
    calc_result: Dict[str, Any],
    reasoning_text: str,
    raw_input_path: str,
    disease_description: str, 
    output_dir: str = None, # 改为 None，便于接收状态机的动态传参
    template_dir: str = "templates"
) -> Dict[str, Any]:
    """
    Word 评估报告自动化渲染工具。
    接收流水线前端传递的所有内存状态数据，自动挑选匹配的边坡模板，渲染并落盘为最终的 Docx 报告。
    """
    if not parsed_data or "basic_info" not in parsed_data:
        return {"error": "缺少基础解析数据，无法生成报告"}

    print("   📝 [Tool 5] 正在渲染并装订最终评估报告...")
    
    # === 核心逻辑 1：动态输出路径接管 ===
    final_output_dir = output_dir if output_dir else "data_io/output_reports"
    os.makedirs(final_output_dir, exist_ok=True)
    
    basic_info = parsed_data.get("basic_info", {})
    score_data = parsed_data.get("score_data", [])
    calc_data = calc_result.get("calculation_result", {})
    
    # === 核心逻辑 2：提取主键与路径信息，优化文件命名 ===
    # 提取输入的文件夹名称（例如 2.K046+272-K046+400右幅）
    folder_name = os.path.basename(os.path.normpath(raw_input_path))
    slope_code = str(basic_info.get("slope_code", "")).strip().upper()
    if not slope_code:
        slope_code = str(basic_info.get("slope_id", "Unknown")).strip().upper()
        
    output_filename = f"{folder_name}_{slope_code}_评估报告.docx"
    output_filepath = os.path.join(final_output_dir, output_filename)
    
    # 2. 准备 Word 模板上下文 (Context)
    context = {**basic_info}
    slope_type = context.get("slope_type", "二元")
    
    # --- 自然语言生成：消灭概况语病 ---
    prot = str(context.get('surface_protection', '无')).strip()
    reinf = str(context.get('surface_reinforcement', '无')).strip()
    prot_str = "无坡面防护" if prot in ['无', '/', 'None', '', 'null'] else f"坡面防护形式为{prot}"
    reinf_str = "无加固形式" if reinf in ['无', '/', 'None', '', 'null'] else f"加固形式为{reinf}"
    context['protection_desc'] = f"{prot_str}，{reinf_str}。"

    surf_drain = str(context.get('surface_drainage', '无')).strip()
    undr_drain = str(context.get('underground_drainage', '无')).strip()
    surf_str = "无地表排水设施" if surf_drain in ['无', '/', 'None', '', 'null'] else f"地表排水设施为{surf_drain}"
    undr_str = "无地下排水设施" if undr_drain in ['无', '/', 'None', '', 'null'] else f"地下排水设施为{undr_drain}"
    context['disease_desc'] = f"{surf_str}，{undr_str}。"

    step_val = context.get("has_inspection_step")
    if isinstance(step_val, bool): 
        context["has_inspection_step"] = "有" if step_val else "无"

    # --- 展平打分列表与提取破坏历史 ---
    history_desc = "该边坡无以往破坏记录。" 
    for score_item in score_data:
        item_code = str(score_item.get("item", "")).upper()
        if not item_code: continue
        
        context[f"{item_code}_weight"] = score_item.get("weight")
        context[f"{item_code}_result"] = score_item.get("result")
        context[f"{item_code}_score"] = score_item.get("score")
        
        if item_code in ["B3", "B4"]:
            res_text = str(score_item.get("result", "")).strip()
            if "有记录" in res_text or "观察到" in res_text:
                match = re.search(r'[（(](.*?)[)）]', res_text)
                if match:
                    history_desc = f"经现场调查及资料核实，该边坡以往曾发生破坏（{match.group(1)}）。"
                else:
                    history_desc = f"经现场调查及资料核实，{res_text}。"
    context['history'] = history_desc

    # --- 融合算力层数据 ---
    context["IS"] = calc_data.get("IS", "/")
    context["CS"] = calc_data.get("CS", "/")
    context["TS"] = calc_data.get("TS", "/")
    context["Level"] = calc_data.get("Level", "/")
    
    # --- 融合大模型数据 ---
    context["stability_level"] = context["Level"]
    context["reasoning"] = reasoning_text if reasoning_text else "/"
    context["suggested_action"] = "建议加强日常巡查力度，保持常规的日常管养与巡视。" if context["Level"] in ["I类", "II类"] else "建议开展专业维修加固，并采取预防性工程处治措施。"
    
    # === 核心逻辑 3：注入图文交叉病害描述 ===
    # 注意此处的键名必须与 Word 模板中的占位符完全一致
    context["disease_description"] = disease_description if disease_description else "无病害描述数据。"
    
    # 3. 匹配模板并渲染
    template_paths = {
        "二元": os.path.join(template_dir, "各边坡风险评估报告样例（二元）.docx"),
        "土质": os.path.join(template_dir, "各边坡风险评估报告样例（土质）.docx"),
        "岩质": os.path.join(template_dir, "各边坡风险评估报告样例（岩质）.docx")
    }
    target_template = next((template_paths[key] for key in template_paths.keys() if key in slope_type), None)
    
    if not target_template or not os.path.exists(target_template):
        return {"error": f"找不到对应的评估模板，边坡类型: {slope_type}"}

    try:
        doc = DocxTemplate(target_template)
        
        # 4. 洗图与现场照片插入模块
        for i in range(1, 11): context[f'image_{i}'] = ""
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        # 如果输入路径本身是一个目录，或者包含在一个目录中，尝试提取同名照片
        input_dir = os.path.dirname(raw_input_path) if os.path.isfile(raw_input_path) else raw_input_path
        if os.path.isdir(input_dir):
            for filename in os.listdir(input_dir):
                name, ext = os.path.splitext(filename)
                if name.startswith("image_") and ext.lower() in valid_exts:
                    img_path = os.path.join(input_dir, filename)
                    try:
                        with Image.open(img_path) as img:
                            clean_img = img.convert("RGB")
                            img_byte_arr = io.BytesIO()
                            clean_img.save(img_byte_arr, format='JPEG')
                            img_byte_arr.seek(0)
                        context[name] = InlineImage(doc, img_byte_arr, width=Mm(140))
                    except Exception as e:
                        print(f"   ⚠️ 图片装载失败 [{filename}]: {e}")

        # 5. 生成落盘
        doc.render(context)
        doc.save(output_filepath)
        return {
            "status": "success",
            "report_path": output_filepath
        }
        
    except Exception as e:
        return {"error": f"Word 渲染器发生致命错误: {str(e)}"}