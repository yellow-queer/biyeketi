"""
RAG 知识库模块
实现文档加载、向量化和相似度检索功能
支持实时更新知识库
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import hashlib


class RAGKnowledgeBase:
    """
    RAG 知识库
    使用 sentence-transformers 进行文档向量化
    使用 FAISS 进行相似度检索
    """
    
    def __init__(self, documents_path: str, index_path: str, embedding_model: str = None):
        """
        初始化知识库
        
        Args:
            documents_path: 文档目录路径
            index_path: 向量索引保存路径
            embedding_model: 嵌入模型名称
        """
        self.documents_path = Path(documents_path)
        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        
        self.documents = []  # 存储文档内容
        self.document_embeddings = None  # 文档向量
        self.model = None  # 嵌入模型
        self.index = None  # FAISS 索引
        
        print(f"📚 RAG 知识库初始化中...")
        print(f"  文档目录：{self.documents_path}")
        print(f"  索引目录：{self.index_path}")
    
    def load_model(self):
        """加载嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  加载嵌入模型：{self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)
            print("  ✓ 模型加载成功")
        except Exception as e:
            print(f"  ✗ 模型加载失败：{e}")
            raise
    
    def load_documents(self) -> List[Dict]:
        """
        加载文档目录下的所有文档
        
        Returns:
            List[Dict]: 文档列表，每个文档包含：
                - content: 文档内容
                - path: 文件路径
                - name: 文件名
                - hash: 内容哈希（用于检测更新）
        """
        documents = []
        supported_extensions = ['.txt', '.md', '.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png']
        
        print(f"\n📄 扫描文档目录：{self.documents_path}")
        
        if not self.documents_path.exists():
            print(f"  ⚠ 文档目录不存在，创建空目录")
            self.documents_path.mkdir(parents=True, exist_ok=True)
            return documents
        
        # 遍历文档目录
        for file_path in self.documents_path.glob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    # 跳过 README 文件
                    if file_path.name.lower() == 'readme.md':
                        continue
                    
                    # 根据文件类型读取内容
                    suffix = file_path.suffix.lower()
                    
                    if suffix in ['.pdf']:
                        content = self._read_pdf(file_path)
                    elif suffix in ['.doc', '.docx']:
                        content = self._read_word(file_path)
                    elif suffix in ['.jpg', '.jpeg', '.png']:
                        content = self._read_image(file_path)
                    else:
                        # 文本文件
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    
                    if not content:
                        print(f"  ⚠ 文件内容为空：{file_path.name}")
                        continue
                    
                    # 计算内容哈希
                    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                    
                    documents.append({
                        'content': content,
                        'path': str(file_path),
                        'name': file_path.name,
                        'hash': content_hash,
                        'type': suffix[1:].upper()
                    })
                    
                    print(f"  ✓ 加载文档：{file_path.name} ({suffix[1:].upper()})")
                    
                except Exception as e:
                    print(f"  ✗ 加载失败 {file_path.name}: {e}")
        
        print(f"  共加载 {len(documents)} 个文档\n")
        return documents
    
    def _read_pdf(self, file_path: Path) -> str:
        """读取 PDF 文件内容"""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(file_path))
            text = []
            for page in reader.pages:
                text.append(page.extract_text())
            return '\n'.join(text)
        except ImportError:
            print(f"  ⚠ 需要安装 PyPDF2: pip install PyPDF2")
            return ""
        except Exception as e:
            print(f"  ⚠ PDF 读取失败：{e}")
            return ""
    
    def _read_word(self, file_path: Path) -> str:
        """读取 Word 文件内容"""
        try:
            suffix = file_path.suffix.lower()
            if suffix == '.docx':
                from docx import Document
                doc = Document(str(file_path))
                text = []
                for para in doc.paragraphs:
                    text.append(para.text)
                return '\n'.join(text)
            elif suffix == '.doc':
                # .doc 需要额外库
                print(f"  ⚠ .doc 格式需要额外配置")
                return ""
            return ""
        except ImportError:
            print(f"  ⚠ 需要安装 python-docx: pip install python-docx")
            return ""
        except Exception as e:
            print(f"  ⚠ Word 读取失败：{e}")
            return ""
    
    def _read_image(self, file_path: Path) -> str:
        """读取图片内容（OCR 识别）"""
        try:
            from paddleocr import PaddleOCR
            import os
            
            # 检查是否已初始化 OCR
            if not hasattr(self, 'ocr'):
                print(f"  初始化 OCR 引擎...")
                self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
            
            # OCR 识别
            img_path = str(file_path)
            result = self.ocr.ocr(img_path, cls=True)
            
            # 提取文字
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    text_lines.append(line[1][0])
            
            content = '\n'.join(text_lines)
            
            if content:
                return f"[图片内容 - {file_path.name}]\n{content}"
            else:
                return ""
                
        except ImportError:
            print(f"  ⚠ 需要安装 PaddleOCR: pip install paddlepaddle paddleocr")
            return ""
        except Exception as e:
            print(f"  ⚠ 图片 OCR 失败：{e}")
            return ""
    
    def create_embeddings(self, documents: List[Dict], batch_size: int = 32):
        """
        创建文档向量
        
        Args:
            documents: 文档列表
            batch_size: 批处理大小
        """
        if not documents:
            print("⚠ 没有文档需要向量化")
            self.documents = []
            self.document_embeddings = None
            return
        
        if self.model is None:
            self.load_model()
        
        print("🔄 创建文档向量...")
        
        # 提取文档内容
        texts = [doc['content'] for doc in documents]
        self.documents = documents
        
        # 批量创建向量
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)
            print(f"  处理进度：{min(i+batch_size, len(texts))}/{len(texts)}")
        
        # 合并向量
        self.document_embeddings = np.vstack(embeddings)
        
        print(f"✓ 完成向量化，向量维度：{self.document_embeddings.shape}")
    
    def build_index(self):
        """构建 FAISS 索引"""
        try:
            import faiss
            
            if self.document_embeddings is None:
                print("⚠ 没有向量数据，无法构建索引")
                return
            
            print("\n🔧 构建 FAISS 索引...")
            
            # 创建索引
            dimension = self.document_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            
            # 添加向量
            self.index.add(self.document_embeddings.astype('float32'))
            
            print(f"✓ 索引构建完成，包含 {self.index.ntotal} 个向量")
            
            # 保存索引
            self.save_index()
            
        except ImportError:
            print("⚠ FAISS 未安装，使用简单的余弦相似度检索")
            self.index = None
    
    def save_index(self):
        """保存索引和文档到磁盘"""
        if not self.index_path.exists():
            self.index_path.mkdir(parents=True, exist_ok=True)
        
        # 保存文档
        doc_file = self.index_path / 'documents.pkl'
        with open(doc_file, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # 保存向量
        if self.document_embeddings is not None:
            vec_file = self.index_path / 'embeddings.npy'
            np.save(vec_file, self.document_embeddings)
        
        # 保存 FAISS 索引
        if self.index is not None:
            import faiss
            index_file = self.index_path / 'faiss.index'
            faiss.write_index(self.index, str(index_file))
        
        print(f"✓ 索引已保存到：{self.index_path}")
    
    def load_index(self):
        """从磁盘加载索引"""
        if not self.index_path.exists():
            print("⚠ 索引目录不存在")
            return False
        
        try:
            # 加载文档
            doc_file = self.index_path / 'documents.pkl'
            if doc_file.exists():
                with open(doc_file, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"✓ 加载 {len(self.documents)} 个文档")
            
            # 加载向量
            vec_file = self.index_path / 'embeddings.npy'
            if vec_file.exists():
                self.document_embeddings = np.load(vec_file)
                print(f"✓ 加载向量：{self.document_embeddings.shape}")
            
            # 加载 FAISS 索引
            index_file = self.index_path / 'faiss.index'
            if index_file.exists():
                import faiss
                self.index = faiss.read_index(str(index_file))
                print(f"✓ 加载 FAISS 索引：{self.index.ntotal} 个向量")
            
            return True
            
        except Exception as e:
            print(f"✗ 加载索引失败：{e}")
            return False
    
    def check_for_updates(self) -> bool:
        """
        检查文档是否有更新
        
        Returns:
            bool: 是否有更新
        """
        if not self.documents:
            return True
        
        current_docs = self.load_documents()
        
        # 比较文档数量和哈希
        if len(current_docs) != len(self.documents):
            return True
        
        for old_doc, new_doc in zip(self.documents, current_docs):
            if old_doc['hash'] != new_doc['hash']:
                return True
        
        return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回最相关的 K 个文档
            
        Returns:
            List[Dict]: 相关文档列表，包含：
                - content: 文档内容片段
                - name: 文档名称
                - score: 相似度分数
        """
        if not self.documents:
            print("⚠ 知识库为空")
            return []
        
        if self.model is None:
            self.load_model()
        
        # 创建查询向量
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        results = []
        
        if self.index is not None:
            # 使用 FAISS 检索
            import faiss
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        'content': doc['content'][:500] + '...' if len(doc['content']) > 500 else doc['content'],
                        'name': doc['name'],
                        'score': float(1 / (1 + distances[0][i]))  # 转换为相似度分数
                    })
        else:
            # 使用余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
            
            # 获取 top_k 个最相似的文档
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                doc = self.documents[idx]
                results.append({
                    'content': doc['content'][:500] + '...' if len(doc['content']) > 500 else doc['content'],
                    'name': doc['name'],
                    'score': float(similarities[idx])
                })
        
        return results
    
    def rebuild_if_needed(self):
        """检查并重建索引（如果有更新）"""
        if self.check_for_updates():
            print("📝 检测到文档更新，重建索引...")
            documents = self.load_documents()
            if documents:
                self.create_embeddings(documents)
                self.build_index()
            else:
                print("⚠ 没有文档需要索引")
        else:
            print("✓ 文档无更新，使用现有索引")


# 创建全局实例
_knowledge_base = None


def get_knowledge_base():
    """获取知识库实例（单例模式）"""
    global _knowledge_base
    if _knowledge_base is None:
        from config import RAG_DOCUMENTS_PATH, RAG_INDEX_PATH, EMBEDDING_MODEL
        _knowledge_base = RAGKnowledgeBase(RAG_DOCUMENTS_PATH, RAG_INDEX_PATH, EMBEDDING_MODEL)
        
        # 尝试加载现有索引
        if _knowledge_base.load_index():
            _knowledge_base.rebuild_if_needed()
        else:
            # 首次使用，创建索引
            _knowledge_base.load_model()
            documents = _knowledge_base.load_documents()
            if documents:
                _knowledge_base.create_embeddings(documents)
                _knowledge_base.build_index()
    
    return _knowledge_base


if __name__ == "__main__":
    # 测试代码
    kb = get_knowledge_base()
    
    query = "柑橘实蝇检测方法"
    print(f"\n🔍 搜索：{query}")
    results = kb.search(query, top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['name']} (相似度：{result['score']:.3f})")
        print(f"   内容：{result['content'][:100]}...")
