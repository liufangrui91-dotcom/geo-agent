import os
import sys
import shutil
import datetime  # 新增：用于生成单次任务的时间戳
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from agent_core.router_workflow import build_workflow
from agent_core.kb_manager import ingest_document_to_chroma  # <--- 新增到这里
from config import API_KEY, BASE_URL


class StreamlitRedirect:
    """劫持 print 输出流并实时渲染到 Streamlit"""
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def write(self, msg):
        if msg.strip():
            self.text += msg + "\n"
            self.container.code(self.text, language="log")
            
    def flush(self): 
        pass

@st.cache_resource
def load_vector_db():
    """
    全局单例：挂载本地 Chroma 向量知识库。
    利用 Streamlit 的缓存机制，保证大模型 Embedding 和数据库仅被实例化一次。
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model="Qwen/Qwen3-Embedding-8B"
    )
    
    db_path = os.path.join(os.getcwd(), "knowledge_base", "chroma_db_qwen")
    if not os.path.exists(db_path):
        return None
        
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    # k=3 代表每次检索返回最相关的3条规范内容
    return vector_db.as_retriever(search_kwargs={"k": 3})

def init_ui():
    st.set_page_config(page_title="Geo-Agent 高风险边坡评估", layout="wide")
    st.title("🏔️ Geo-Agent 边坡智能评估系统")
    st.markdown("基于 LangGraph 工作流，支持图文双模态输入，自动生成标准勘察报告。")

    # 全局知识库挂载
    retriever = load_vector_db()

    with st.sidebar:
        st.header("⚙️ 引擎状态")
        st.success("工具就绪")
        st.success("LangGraph 路由中枢就绪")
        if retriever:
            st.success("📚 Chroma 规范知识库已挂载")
        else:
            st.error("❌ 找不到 Chroma 知识库，Tool 4 将降级为无参考推理")

    # ... 上方的 sidebar 代码保持不变 ...

    # 声明三个独立的标签页
    tab_single, tab_batch, tab_kb = st.tabs(["📄 单次上传评估", "📁 批量目录扫描", "📚 知识库管理"])

    # ==========================================
    # 标签页 1：单次图文混合上传
    # ==========================================
    with tab_single:
        st.subheader("📤 临时数据源上传")
        uploaded_files = st.file_uploader(
            "请上传该边坡的勘察记录（图片/文本）及全景图（风景图务必以 image_ 开头命名）：", 
            type=['png', 'jpg', 'jpeg', 'txt'], 
            accept_multiple_files=True
        )
        
        output_dir_single = st.text_input(
            "报告输出绝对路径（留空默认存入 data_io/output_reports）：", 
            key="out_single"
        )

        if st.button("🚀 启动单次评估", key="btn_single"):
            if not uploaded_files:
                st.error("阻断：请先上传基础勘察资料。")
            else:
                st.divider()
                st.subheader("🔄 智能体工作流进程")
                
                # 1. 动态创建时间戳临时文件夹
                timestamp = datetime.datetime.now().strftime("%Y%md_%H%M%S")
                temp_run_dir = os.path.join("data_io", "temp_inputs", f"run_{timestamp}")
                os.makedirs(temp_run_dir, exist_ok=True)
                
                # 2. 将前端上传的文件落盘到临时目录
                for uf in uploaded_files:
                    file_path = os.path.join(temp_run_dir, uf.name)
                    with open(file_path, "wb") as f:
                        f.write(uf.getbuffer())
                st.success(f"临时环境就绪，资料已存入: {temp_run_dir}")
                
                # 3. 配置状态机并启动
                app = build_workflow(retriever=retriever)
                initial_state = {
                    "raw_input_path": temp_run_dir,
                    "output_dir": output_dir_single.strip() if output_dir_single.strip() else None
                }
                
                log_container = st.empty()
                original_stdout = sys.stdout
                sys.stdout = StreamlitRedirect(log_container)
                
                try:
                    for output in app.stream(initial_state):
                        for node_name, node_state in output.items():
                            print(f"✅ 完成节点: {node_name}")
                            if "disease_description" in node_state and node_state["disease_description"]:
                                st.info(f"【图文交叉感知结果】\n{node_state['disease_description']}")
                            if "calculation_results" in node_state:
                                calc_data = node_state["calculation_results"].get("calculation_result", {})
                                st.metric(label="定级硬计算 (TS评分)", value=calc_data.get("TS", "N/A"), delta=calc_data.get("Level", ""))
                            if "report_path" in node_state:
                                st.success(f"报告渲染完成！路径: {node_state['report_path']}")
                except Exception as e:
                    st.error(f"工作流异常中断: {str(e)}")
                finally:
                    sys.stdout = original_stdout


    # ==========================================
    # 标签页 2：批量目录扫描 (保留原有逻辑)
    # ==========================================
    with tab_batch:
        st.subheader("📁 批量数据源设置")
        base_data_dir = st.text_input(
            "请输入包含多个边坡文件夹的【本地根目录绝对路径】：", 
            key="in_batch"
        )
        
        output_base_dir = st.text_input(
            "报告输出绝对路径（留空默认存入 data_io/output_reports）：", 
            key="out_batch"
        )

        if st.button("🚀 启动批量评估", key="btn_batch"):
            if not base_data_dir or not os.path.isdir(base_data_dir):
                st.error("阻断：请输入有效的本地目录路径。")
            else:
                slope_folders = [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]
                if not slope_folders:
                    st.warning("指定目录下没有找到子文件夹。")
                else:
                    st.divider()
                    st.info(f"扫描完毕，共发现 {len(slope_folders)} 个边坡数据包，准备启动批量评估。")
                    
                    app = build_workflow(retriever=retriever)
                    
                    for folder_name in slope_folders:
                        folder_path = os.path.join(base_data_dir, folder_name)
                        st.markdown(f"### ▶ 正在评估: `{folder_name}`")
                        
                        initial_state = {
                            "raw_input_path": folder_path,
                            "output_dir": output_base_dir.strip() if output_base_dir.strip() else None
                        }
                        
                        log_container = st.empty()
                        original_stdout = sys.stdout
                        sys.stdout = StreamlitRedirect(log_container)
                        
                        try:
                            for output in app.stream(initial_state):
                                for node_name, node_state in output.items():
                                    print(f"✅ 完成节点: {node_name}")
                                    if "disease_description" in node_state and node_state["disease_description"]:
                                        st.info(f"【图文交叉感知结果】\n{node_state['disease_description']}")
                                    if "calculation_results" in node_state:
                                        calc_data = node_state["calculation_results"].get("calculation_result", {})
                                        st.metric(label="定级硬计算 (TS评分)", value=calc_data.get("TS", "N/A"), delta=calc_data.get("Level", ""))
                                    if "report_path" in node_state:
                                        st.success(f"[{folder_name}] 报告渲染完成！路径: {node_state['report_path']}")
                        except Exception as e:
                            st.error(f"工作流异常中断: {str(e)}")
                        finally:
                            sys.stdout = original_stdout

    # ==========================================
    # 标签页 3：知识库管理 (占位)
    # ==========================================
    # ==========================================
    # 标签页 3：知识库管理 (动态 RAG 注入)
    # ==========================================
    with tab_kb:
        st.subheader("📚 规范向量知识库动态接入")
        st.info("上传最新的工程规范、地质勘察标准（PDF/Word），系统将自动解析并更新底层大模型知识库，立刻生效于后续的 Tool 4 专家推理中。")
        
        kb_file = st.file_uploader("请选择需要入库的规范文件：", type=['pdf', 'docx'])
        
        if st.button("📥 开始解析并入库", key="btn_kb_upload"):
            if kb_file is None:
                st.warning("请先上传待入库的文件。")
            else:
                with st.spinner("🧠 正在解析文档并执行高维向量化计算，请稍候..."):
                    # 1. 暂存上传的文件
                    temp_kb_dir = os.path.join("data_io", "temp_kb")
                    os.makedirs(temp_kb_dir, exist_ok=True)
                    temp_path = os.path.join(temp_kb_dir, kb_file.name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(kb_file.getbuffer())
                    
                    # 2. 调用核心入库算子
                 
                    # 确保路径与之前你设定的 Chroma_db 路径严格一致
                    db_dir = os.path.join(os.getcwd(), "knowledge_base", "chroma_db_qwen") 
                    
                    success, msg = ingest_document_to_chroma(temp_path, db_dir)
                    
                    # 3. 结果反馈
                    if success:
                        st.success(f"✅ {msg}")
                        st.balloons() # 庆祝一下知识库扩展成功
                    else:
                        st.error(f"❌ 注入失败: {msg}")

if __name__ == "__main__":
    init_ui()