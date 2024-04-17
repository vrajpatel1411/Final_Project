from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.document_loaders.csv_loader import CSVLoader

metadata_field_info = [
    AttributeInfo(
        name="Brand",
        description="The name of Brand such as Samsung, Hp, Acer, Lenevo and many other",
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


loader = CSVLoader(file_path='final_dataset.csv')

documents = loader.load()
print(len(documents))
print(documents[0])


