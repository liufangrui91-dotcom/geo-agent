import os
import fitz  # PyMuPDF
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import API_KEY, BASE_URL

def extract_text_from_file(file_path: str) -> str:
    """依据文件后缀自动调用对应的解析引擎提取纯文本"""
    ext = os.path.splitext(file_path)[-1].lower()
    text = ""
    
    try:
        if ext == ".pdf":
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        elif ext in [".docx", ".doc"]:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            return ""
    except Exception as e:
        print(f"解析文档异常: {str(e)}")
        
    return text

def ingest_document_to_chroma(file_path: str, chroma_db_path: str) -> tuple[bool, str]:
    """核心注入流水线：解析 -> 切片 -> 向量化 -> 入库"""
    print(f"开始处理知识库入库: {os.path.basename(file_path)}")
    
    # 1. 文本抽取
    raw_text = extract_text_from_file(file_path)
    if not raw_text.strip():
        return False, "未能从文档中提取到有效文本，或文件格式暂不支持。"

    # 2. 文本切片 (Chunking) - 保留一定的上下文重叠度
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    )
    chunks = text_splitter.split_text(raw_text)
    
    if not chunks:
        return False, "文本切片失败。"

    # 3. 装载 Embedding 模型与数据库连接
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            model="Qwen/Qwen3-Embedding-8B"
        )
        
        vector_db = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
        
        # 4. 写入元数据并追加至数据库
        metadatas = [{"source": os.path.basename(file_path)} for _ in chunks]
        vector_db.add_texts(texts=chunks, metadatas=metadatas)
        
        return True, f"成功！提取并向量化了 {len(chunks)} 个规范知识块。"
    except Exception as e:
        return False, f"向量化入库阶段发生异常: {str(e)}"