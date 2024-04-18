from langchain.chains.llm import LLMChain

import Bert
import NER
import Ensemble_retriever
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain import hub
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
class ResponseGenerator:

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key="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")
        self.intent_classifier = Bert.BertClassification()
        self.ner = NER.NamedEntityRecognizer()
        self.ensemble=Ensemble_retriever.prepare_dataset()
        self.fixed_entities=dict()
        self.chat_history=[]

    def getResponse(self, user_query):
        intents=self.intent_classifier.get_prediction(user_query)
        label=intents[0]['label']

        intent_handlers = {
            'get_refund': self.handle_refund,
            'cancel_order': self.handle_order_cancellation,
            'contact_human': self.handle_human_contact,
            'payment_issue': self.handle_payment_issue,
            'recover_password': self.handle_password_recovery,
            'farewell': self.handle_farewell,
            'track_order': self.handle_order_tracking,
            'search_product_information': self.handle_product_information_search,
            'unknown': self.handle_unknown_request,
            'welcome': self.handle_welcome,
            'search': self.handle_search
        }
        handler = intent_handlers.get(label, self.handle_unknown_request)
        response= handler(user_query)
        return response

    def handle_refund(self, input):
        get_entity=self.ner.predict_entities()
        return "Refund Handler"
    # Your code to handle refund request

    def handle_order_cancellation(self,input):
        return "Order Cancellation Handler"
    # Your code to handle order cancellation

    def handle_human_contact(self,input):
        return "Human Contact Handler"
    # Your code to contact a human

    def handle_payment_issue(self,input):
        return "Payment Issue Handler"
    # Your code to handle payment issues

    def handle_password_recovery(self,input):
        return "Password Recovery Handler"
    # Your code to recover password

    def handle_farewell(self,input):
        return "Fairwell Handler"
    # Your code to bid farewell

    def handle_order_tracking(self,input):
        return "Order Tracking Handler"
    # Your code to track order

    def handle_product_information_search(self,input):
        labels = [ "category","price range","brand","model name"]
        entities = self.ner.predict_entities(input, labels)
        temp=dict()
        for entity in entities:
            if(entity['score']>0.50):
                if(entity['label'] not in temp):
                    temp.update({entity['label']:entity["text"]})
                else:
                    t=temp[entity['label']]
                    s=t+" "+entity["text"]
                    temp[entity['label']]=s

        for key in temp:
            self.fixed_entities[key] = temp[key]
        question_string = str(self.fixed_entities)
        final_question = input + " entities:" + question_string
        prompt_template = """You are Electronic genius and you work as a sales person in electronic shop for last 20 
        years.Your task is to take the user input and context return by the retrieval and frame the response that 
        helps user. Remember to give concise response to user query. Also, take into account past chat history when 
        providing the response. If you have any confusion regarding the product asked further questions. Also, 
        used your experience when providing the response. Do not give any bad response. Finally, if you asking any 
        further question then boolean to starting of the response.
             
        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        combine_docs_chain = create_stuff_documents_chain(
            self.llm,qa_prompt
        )
        retrieval_chain = create_retrieval_chain(self.ensemble, combine_docs_chain)
        results = retrieval_chain.invoke({"input": final_question,"chat_history":self.chat_history})
        self.chat_history.append([HumanMessage(content=final_question), results["answer"]])
        return results['answer']
    # Your code to search for product information

    def handle_unknown_request(self,input):
        return "Unknown Request Handler"
    # Your code to handle unknown requests

    def handle_welcome(self,input):
        prompt_template = """
        
                You are Electronic genius and you work as a sales person in electronic shop for last 20 
                years.Your task is to take the user input, and response a user-friendly greeting response with small introduction of yourself.
                 Remember to give concise response to user query. Also, take into account past chat history when 
                providing the response. If you have any confusion regarding the product asked further questions. Also, 
                used your experience when providing the response. Do not give any bad response. Finally, if you asking any 
                further question then boolean to starting of the response.
                """
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        simple_chain=LLMChain(llm=self.llm, prompt=qa_prompt)
        results = simple_chain.invoke({"input": input, "chat_history": self.chat_history})
        print(results)
        self.chat_history.append([HumanMessage(content=input), results["text"]])
        return results['text']
    # Your code to welcome the user

    def handle_search(self,input):
        labels=[ "category","range","brand","model name"]
        entities=self.ner.predict_entities(input,labels)
        temp = dict()
        for entity in entities:
            if (entity['score'] > 0.5):
                if (entity['label'] not in temp):
                    temp.update({entity['label']: entity["text"]})
                else:
                    t = temp[entity['label']]
                    s = t + " " + entity["text"]
                    temp[entity['label']] = s
        print(temp)
        for key in temp:
            print(key)
            self.fixed_entities[key] = temp[key]
        question_string=str(self.fixed_entities)
        final_question = input + " entities:" + question_string
        # prompt_template = """You are Electronic genius and you work as a sales person in electronic shop for last 20
        #         years.Your task is to take the user input and context return by the retrieval and frame the response that
        #         helps user. Remember to give concise response to user query. Also, take into account past chat history when
        #         providing the response. If you have any confusion regarding the product asked further questions. Also,
        #         used your experience when providing the response. Do not give any bad response. Finally, if you asking any
        #         further question then boolean to starting of the response.
        #         User Input : {input}
        #         Chat History : {chat_history}
        #         Retrieved Document: {context}"""
        # qa_prompt = PromptTemplate(
        #     input_variables=["input","chat_history","context"],
        #     template=prompt_template,
        # )
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, retrieval_qa_chat_prompt
        )
        retrieval_chain = create_retrieval_chain(self.ensemble, combine_docs_chain)
        results = retrieval_chain.invoke({"input": final_question, "chat_history": self.chat_history})
        self.chat_history.append([HumanMessage(content=final_question), results["answer"]])
        return results['answer']
