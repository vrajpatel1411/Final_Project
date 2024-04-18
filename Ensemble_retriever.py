from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain.document_loaders.csv_loader import CSVLoader
# from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai import ChatOpenAI
from langchain.retrievers.self_query.chroma import ChromaTranslator
import nltk
from langchain import hub
from langchain.retrievers import EnsembleRetriever


retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

nltk.download("punkt")

metadata_field_info = [
    AttributeInfo(
        name="Brand",
        description="The name of Brand such as Samsung, Hp, Acer, Lenovo and many other",
        type="string",
    ),
    AttributeInfo(
        name="Color",
        description="The Color of the device or electronic item",
        type="string",
    ),
    AttributeInfo(
        name="Hard Disk Size",
        description="The Hard disk size is the hard disk space in that device",
        type="float",
    ),
    AttributeInfo(
        name="Memory Storage Capacity",
        description="The Memory storage capacity is the memory space in that device",
        type="float",
    ),
    AttributeInfo(
        name="Model Name",
        description="The name of the model",
        type="float",
    ),
    AttributeInfo(
        name="Ram Memory Installed Size",
        description="Ram size of the device. It is in gb",
        type="float",
    ),
    AttributeInfo(
        name="Screen Size",
        description="Screen Size of the device",
        type="float",
    ),
    AttributeInfo(
        name="price",
        description="The price of the device",
        type="float",
    ),
]

document_content_description = "Description of mobile and laptop device"

class prepare_dataset():

    def __init__(self):
        self.ensemble_retriever = None
        self.llm = ChatOpenAI(temperature=0.5, openai_api_key="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")
        self. embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        self.vectordb = Chroma(persist_directory="embeddings\\", embedding_function=self.embedding_function)


    def build_ensemble_retriever(self):
        query_retriever = SelfQueryRetriever.from_llm(
            self.llm,
            self.vectordb,
            document_content_description,
            metadata_field_info,
            structured_query_translator=ChromaTranslator(),
            search_kwargs={"k":10},
            verbose=True
        )
        similarity_retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 10}
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[similarity_retriever,query_retriever], weights=[0.5,0.5]
        )

        # combine_docs_chain = create_stuff_documents_chain(
        #     self.llm, retrieval_qa_chat_prompt
        # )

        # retrieval_chain = create_retrieval_chain(self.ensemble_retriever, combine_docs_chain)

        return self.ensemble_retriever
        # return retrieval_chain






    # loader = CSVLoader(file_path='final_dataset.csv',
    #                    metadata_columns=['title', 'cleaned_price', 'Brand', 'Color', 'Model Name',
    #                                      'Ram Memory Installed Size', 'Screen Size', 'Hard Disk Size',
    #                                      'Memory Storage Capacity'],autodetect_encoding=True)
    #
    # self.documents = loader.load()

    # self.change_metadata_to_float()
    # self.change_metadata_name()
    # self.cleaned_page_content()
    # self.remove_empty_metadata()
    # self.split_page_content()

    # def change_metadata_to_float(self):
    #     print("Starting Metadata type change")
    #     metadata_columns = ['cleaned_price', 'Ram Memory Installed Size', 'Screen Size', 'Hard Disk Size',
    #                         'Memory Storage Capacity']
    #     for doc in self.documents:
    #         for key in metadata_columns:
    #             if key in doc.metadata.keys() and doc.metadata[key] not in ('', float('nan'), None):
    #                 # print(doc)
    #                 doc.metadata[key] = float(doc.metadata[key])
    #     print("Metadata type changed")
    #
    # def change_metadata_name(self):
    #     print("Change metadata name")
    #     for doc in self.documents:
    #         if "cleaned_price" in doc.metadata.keys():
    #             doc.metadata['price'] = doc.metadata.pop('cleaned_price')
    #     print("metdata name change")
    # def cleaned_page_content(self):
    #     print("Cleaning page content")
    #     for doc in self.documents:
    #         tokens = word_tokenize(doc.page_content.lower())
    #         doc.page_content = ' '.join(tokens)
    #     print("Page Content Cleaned")
    # def remove_empty_metadata(self):
    #     print("Removing empty metadata")
    #     metadata_columns = ['title', 'price', 'Brand', 'Color', 'Model Name', 'Ram Memory Installed Size',
    #                         'Screen Size', 'Hard Disk Size', 'Memory Storage Capacity']
    #     for doc in self.documents:
    #         # Create a new dictionary to store filtered metadata
    #         filtered_metadata = {}
    #         for key in metadata_columns:
    #             # Check if the key exists and its value is not empty, NaN, or None
    #             if key in doc.metadata and doc.metadata[key] not in ('', float('nan'), None):
    #                 # Add the key-value pair to the filtered metadata dictionary
    #                 filtered_metadata[key] = doc.metadata[key]
    #         # Replace the metadata dictionary with the filtered one
    #         doc.metadata = filtered_metadata
    #     print("Removed empty metadata")
    # def split_page_content(self):
    #     print("Split is starting")
    #     text_splitter = CharacterTextSplitter(chunk_size=768, chunk_overlap=20)
    #     self.docs = text_splitter.split_documents(self.documents)
    #     print("Spliting is done")

    # def upload_to_chroma(self):
    #     print("Starting embedding function")
    #     embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    #     print("Embedding function intialized")
    #     print("Adding documents to chroma")
    #     vectordb = Chroma(persist_directory="embeddings\\",embedding_function=embedding_function)
    #     # vectordb.add_documents(self.docs)
    #     # print(vectordb._collection.count())
    #     # db = Chroma.from_documents(self.docs, embedding_function)
    #     return vectordb