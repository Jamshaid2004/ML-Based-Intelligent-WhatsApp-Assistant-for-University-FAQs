from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

from source.schemas import FAQAIResponse

load_dotenv()

class RAGFAQBot:
    """RAG-powered FAQ Bot for WhatsApp integration"""
    
    def __init__(
        self, 
        csv_path: str = "data/synthetic_faq_dataset.csv",
        persist_directory: str = "./chroma_db",
        use_cache: bool = True
    ):
        self.csv_path = csv_path
        self.persist_directory = persist_directory
        self.use_cache = use_cache
        
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.structured_llm = self.llm.with_structured_output(FAQAIResponse)
        
        self.vectorstore = None
        self.retriever = None
        self.faq_df = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize or load the vector store"""
        persist_path = Path(self.persist_directory)
        
        if self.use_cache and persist_path.exists() and any(persist_path.iterdir()):
            print("📦 Loading cached vector store...")
            self._load_vectorstore()
        else:
            print("🆕 Creating new vector store...")
            self._create_vectorstore()
    
    def _load_csv_data(self):
        """Load CSV data"""
        self.faq_df = pd.read_csv(self.csv_path)
        return self.faq_df
    
    def _create_documents(self):
        """Create documents from CSV"""
        df = self._load_csv_data()
        documents = []
        
        for idx, row in df.iterrows():
            content = f"""
Intent: {row['Intent']}
Question: {row['Question']}
Answer: {row['Answer']}
""".strip()
            
            doc = Document(
                page_content=content,
                metadata={
                    'intent': row['Intent'],
                    'question': row['Question'],
                    'answer': row['Answer'],
                }
            )
            documents.append(doc)
        
        return documents
    
    def _create_vectorstore(self):
        """Create vector store"""
        documents = self._create_documents()
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="faq_collection"
        )
        self._setup_retriever()
    
    def _load_vectorstore(self):
        """Load existing vector store"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="faq_collection"
        )
        self._setup_retriever()
    
    def _setup_retriever(self):
        """Setup retriever"""
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )
    
    def _format_context(self, docs):
        """Format documents"""
        formatted = []
        for doc in docs:
            formatted.append(f"""
Intent: {doc.metadata['intent']}
Q: {doc.metadata['question']}
A: {doc.metadata['answer']}
""".strip())
        return "\n\n".join(formatted)
    
    async def query(self, question: str, return_sources: bool = False):
        """Query the RAG system"""
        # Retrievers are runnables, use ainvoke for async
        retrieved_docs = await self.retriever.ainvoke(question)
        
        rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful university FAQ assistant for WhatsApp.

Retrieved FAQs:
{context}

User Question: {question}

Instructions:
1. Identify the correct intent category
2. Provide a clear, concise answer (2-3 sentences max)
3. Be conversational and friendly
4. Use information from the FAQs
5. If uncertain, suggest contacting the university

Format your response for WhatsApp messaging - keep it brief and helpful.
""")
        
        rag_chain = (
            {
                "context": lambda x: self._format_context(retrieved_docs),
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | self.structured_llm
        )
        
        response = await rag_chain.ainvoke(question)
        
        if return_sources:
            return {
                "response": response,
                "sources": [
                    {
                        "intent": doc.metadata['intent'],
                        "question": doc.metadata['question'],
                        "answer": doc.metadata['answer']
                    }
                    for doc in retrieved_docs
                ]
            }
        
        return response
    
    def get_all_intents(self):
        """Get all intent categories"""
        if self.faq_df is None:
            self._load_csv_data()
        return self.faq_df['Intent'].unique().tolist()