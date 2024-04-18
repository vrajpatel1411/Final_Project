import json

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory

import Bert
import NER
import Ensemble_retriever
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain import hub

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")


class ResponseGenerator:

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key="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")
        self.intent_classifier = Bert.BertClassification()
        self.ner = NER.NamedEntityRecognizer()
        self.ensemble = Ensemble_retriever.prepare_dataset()
        self.fixed_entities = dict()
        self.chat_history = []
        self.chat_length=0
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            max_len=500,
            return_messages=True,
        )

    def getResponse(self, user_query):

        intents = self.intent_classifier.get_prediction(user_query)
        label = intents[0]['label']
        if(self.chat_length>0 and self.chat_history[self.chat_length-1][1]):
            response=self.handleDirectQuestion(user_query)
            return response
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
        response = handler(user_query)
        return response

    def handle_refund(self, input):
        get_entity = self.ner.predict_entities()
        return "Refund Handler"

    # Your code to handle refund request

    def handle_order_cancellation(self, input):
        return "Order Cancellation Handler"

    # Your code to handle order cancellation

    def handle_human_contact(self, input):
        return "Human Contact Handler"

    # Your code to contact a human

    def handle_payment_issue(self, input):
        return "Payment Issue Handler"

    # Your code to handle payment issues

    def handle_password_recovery(self, input):
        return "Password Recovery Handler"

    # Your code to recover password

    def handle_farewell(self, input):
        return "Fairwell Handler"

    # Your code to bid farewell

    def handle_order_tracking(self, input):
        return "Order Tracking Handler"

    # Your code to track order

    def handle_product_information_search(self, input):
        print("Handle Product Information Search Handler")
        final_question = input
        prompt_template = """You've been an electronic salesperson for 20 years, known for your expertise. Your task 
        is to provide helpful responses based on user input, chat history and context. If multiple products match the 
        query, offer all options. Consider past chat history and ask further questions if needed. Your responses must 
        be concise and avoid bad answers. Use your experience wisely. Responses should be in JSON format. Your aim is 
        to assist users respectfully and honestly, refraining from harmful or incorrect information. Remember to 
        provide number bullets when response contains multiple options. Strictly speaking when necessary please list 
        down available option to the user. Your Output must Contains isQuestion and response. If user don't specify any requirements just summarize the response.
                        User Input : {input} 
                        Chat History : {chat_history} 
                        Retrieved Document: {context}
                        
                     << Example 1. >>
                        
                        Data Response
                        
                        
                            
                              "input": "Can you show me mobile phones?",
                              "answer": [
                                "isQuestion": true,
                                "response": "Which mobile phone or specific model are you looking for?"
                              ],
                              "explanation": "A question is asked to clarify the user's needs, thus isQuestion is True."
                      
                        
                        << Example 2. >>
                        
                        Data Response
                        
                       
                              "input": "Can you provide me the phone which is black in color?",
                              "answer": [
                                "isQuestion": true,
                                "response": "I have multiple options for black mobile phones. Here are a few: 
                                1. 
                                2. 
                                3. 
                                
                                and so on"
                              ],
                              "explanation": "Multiple options are provided to choose from, hence isQuestion is True."
                         
                        
                        << Example 3. >>
                        
                        Data Response
                        
                    
    
                              "input": "Price of Samsung Galaxy S20 FE 5G?",
                              "answer": [
                                "isQuestion": false,
                                "response": "The price of the Samsung Galaxy S20 FE 5G is $399.99."
                              ],
                              "explanation": "Direct response is given without further questions, so isQuestion is False."
                   
                        
                        << Example 4. >>
                        
                     
                   
                              "input": "Which laptop is best for gaming?",
                              "answer": [
                                "isQuestion": true,
                                "response": "There are several great options for gaming laptops. Are you looking for a specific brand or price range?"
                              ,]
                              "explanation": "Asking for further details to provide tailored recommendations, hence isQuestion is True."
                       
                    
                        
                        """
        qa_prompt = PromptTemplate(
            input_variables=["input", "chat_history", "context"],
            template=prompt_template,
        )

        combine_docs_chain = create_stuff_documents_chain(
            self.llm, qa_prompt
        )
        retrieval_chain = create_retrieval_chain(self.ensemble.build_ensemble_retriever(), combine_docs_chain)
        results = retrieval_chain.invoke({"input": final_question, "chat_history": self.chat_history})


        data = None
        if (results['answer']):
            data = json.loads(results['answer'])
            print(data)
        if "answer" in data:
            self.chat_history.append([HumanMessage(content=final_question), data['answer']["isQuestion"],
                                      AIMessage(content=data['answer']["response"])])
            self.chat_length = self.chat_length + 1
            return data['answer']['response']
        elif "isQuestion" in data and "response" in data:
            self.chat_history.append([HumanMessage(content=final_question), data["isQuestion"],
                                      AIMessage(content=data["response"])])
            self.chat_length = self.chat_length + 1
        else:
            return data
        return data["response"]

    # Your code to search for product information

    def handle_unknown_request(self, input):
        return "Unknown Request Handler"

    # Your code to handle unknown requests

    def handle_welcome(self, input):
        prompt_template = """

                You are Electronic genius and you work as a sales person in electronic shop for last 20
                years.Your task is to take the user input, and response a user-friendly greeting response with small introduction of yourself.
                Remember to give concise response to user query. Also, take into account past chat history when
                providing the response. If you have any confusion regarding the product asked further questions. Also,
                used your experience when providing the response. Do not give any bad response. 
                
                Input Question : {input}
                Chat History:{chat_history}
                """
        qa_prompt = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=prompt_template,
        )
        simple_chain = LLMChain(llm=self.llm, prompt=qa_prompt)
        results = simple_chain.invoke({"input": input, "chat_history": self.chat_history})
        self.chat_history.append([HumanMessage(content=input),False, results["text"]])
        self.chat_length = self.chat_length + 1
        return results['text']

    def handle_search(self, input):
        print("Handling search request")
        labels = ["category", "range", "brand", "model name"]
        entities = self.ner.predict_entities(input, labels)
        temp = dict()
        final_question = input
        prompt_template = """You've been an electronic salesperson for 20 years, known for your expertise. Your task 
        is to provide helpful responses based on user input, chat history and context. If multiple products match the 
        query, offer all options. Consider past chat history and ask further questions if needed. Your responses must 
        be concise and avoid bad answers. Use your experience wisely. Summarize the product information by including it's basic functionality such as key features, specifications, pricing, and suitability for various computing needs such as work, study, entertainment, and light gaming. Responses should be in JSON format. Your aim is 
        to assist users respectfully and honestly, refraining from harmful or incorrect information. Remember to 
        provide number bullets when response contains multiple options. Strictly speaking when necessary please list 
        down available option to the user. Your Output must Contains isQuestion and response. If user don't specify any requirements just summarize the response.
 
                        User Input : {input} 
                        Chat History : {chat_history} 
                        Retrieved Document: {context}
                        
                     << Example 1. >>
                        
                        Data Response
                        
                        
                            
                              "input": "Can you show me mobile phones?",
                              "output": [
                                "isQuestion": true,
                                "response": "Which mobile phone or specific model are you looking for?"
                              ],
                              "explanation": "A question is asked to clarify the user's needs, thus isQuestion is True."
                      
                        
                        << Example 2. >>
                        
                        Data Response
                        
                       
                              "input": "Can you provide me the phone which is black in color?",
                              "output": [
                                "isQuestion": true,
                                "response": "I have multiple options for black mobile phones. Here are a few: 1. 
                                2. 
                                3. 
                                
                                and so on"
                              ],
                              "explanation": "Multiple options are provided to choose from, hence isQuestion is True."
                         
                        
                        << Example 3. >>
                        
                        Data Response
                        
                    
    
                              "input": "Price of Samsung Galaxy S20 FE 5G?",
                              "output": [
                                "isQuestion": false,
                                "response": "The price of the Samsung Galaxy S20 FE 5G is $399.99."
                              ],
                              "explanation": "Direct response is given without further questions, so isQuestion is False."
                   
                        
                        << Example 4. >>
                        
                     
                   
                              "input": "Which laptop is best for gaming?",
                              "output": [
                                "isQuestion": true,
                                "response": "There are several great options for gaming laptops. Are you looking for a specific brand or price range?"
                              ,]
                              "explanation": "Asking for further details to provide tailored recommendations, hence isQuestion is True."
                       
                    
                        
                        """
        qa_prompt = PromptTemplate(
            input_variables=["input", "chat_history", "context"],
            template=prompt_template,
        )
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, qa_prompt
        )
        retrieval_chain = create_retrieval_chain(self.ensemble.build_ensemble_retriever(), combine_docs_chain)
        results = retrieval_chain.invoke({"input": final_question, "chat_history": self.chat_history})
        # print(results)
        data=None
        if(results['answer']):
            data = json.loads(results['answer'])
        if "isQuestion" in data and "response" in data:
            self.chat_history.append([HumanMessage(content=final_question),data["isQuestion"], AIMessage(content=["response"])])
            self.chat_length = self.chat_length + 1
        else:
            return data
        return data["response"]

    def handleDirectQuestion(self, user_query):
        print("Handle DirectQuestion")
        prompt_template1 = """

                        You've been an electronic salesperson for 20 years, known for your expertise. Your task 
        is to provide helpful responses based on user input, chat history and context. If multiple products match the 
        query, offer all options. Consider past chat history and ask further questions if needed. Your responses must 
        be concise and avoid bad answers. Use your experience wisely. Summarize the product information by including it's basic functionality such as key features, specifications, pricing, and suitability for various computing needs such as work, study, entertainment, and light gaming. Responses should be in JSON format. Your aim is 
        to assist users respectfully and honestly, refraining from harmful or incorrect information. Remember to 
        provide number bullets when response contains multiple options. Strictly speaking when necessary please list 
        down available option to the user. Your Output must Contains isQuestion and response. If user don't specify any requirements just summarize the response.

            Input Question: {input}
            Chat History: {chat_history}

Based on the user's response to the previous response of the Ai or chatbot, generate a new question to further assist them in finding the right product or solution. User input may include a selected option from the previous options provided by you. 
                        """
        qa_prompt1 = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=prompt_template1,
        )
        simple_chain = LLMChain(llm=self.llm, prompt=qa_prompt1)
        results = simple_chain.invoke({"input": input, "chat_history": self.chat_history})
        final_question=results["text"]
        print(final_question)
        prompt_template2 = """You've been an electronic salesperson for 20 years, known for your expertise. Your task 
        is to provide helpful responses based on user input, chat history and context. If multiple products match the 
        query, offer all options. Consider past chat history and ask further questions if needed. Your responses must 
        be concise and avoid bad answers. Use your experience wisely. Summarize the product information by including it's basic functionality such as key features, specifications, pricing, and suitability for various computing needs such as work, study, entertainment, and light gaming. Responses should be in JSON format. Your aim is 
        to assist users respectfully and honestly, refraining from harmful or incorrect information. Remember to 
        provide number bullets when response contains multiple options. Strictly speaking when necessary please list 
        down available option to the user. Your Output must Contains isQuestion and response. If user don't specify any requirements just summarize the response.
                               User Input : {input} 
                               Chat History : {chat_history} 
                               Retrieved Document: {context}
                            << Example 1. >>
                            Data Response
                                     "input": "Can you show me mobile phones?",
                                     "output": [
                                       "isQuestion": true,
                                       "response": "Which mobile phone or specific model are you looking for?"
                                     ],
                                     "explanation": "A question is asked to clarify the user's needs, thus isQuestion is True."
                            << Example 2. >>
                            Data Response
                                     "input": "Can you provide me the phone which is black in color?",
                                     "output": [
                                       "isQuestion": true,
                                       "response": "I have multiple options for black mobile phones. Here are a few: 1. 
                                       2. 
                                       3.   and so on"
                                     ],
                                     "explanation": "Multiple options are provided to choose from, hence isQuestion is True."


                               << Example 3. >>

                               Data Response



                                     "input": "Price of Samsung Galaxy S20 FE 5G?",
                                     "output": [
                                       "isQuestion": false,
                                       "response": "The price of the Samsung Galaxy S20 FE 5G is $399.99."
                                     ],
                                     "explanation": "Direct response is given without further questions, so isQuestion is False."


                               << Example 4. >>



                                     "input": "Which laptop is best for gaming?",
                                     "output": [
                                       "isQuestion": true,
                                       "response": "There are several great options for gaming laptops. Are you looking for a specific brand or price range?"
                                     ,]
                                     "explanation": "Asking for further details to provide tailored recommendations, hence isQuestion is True."



                               """
        qa_prompt2 = PromptTemplate(
            input_variables=["input", "chat_history", "context"],
            template=prompt_template2,
        )
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, qa_prompt2
        )
        retrieval_chain = create_retrieval_chain(self.ensemble.build_ensemble_retriever(), combine_docs_chain)
        results = retrieval_chain.invoke({"input": final_question, "chat_history": self.chat_history})


        # print(results)
        data = None
        if (results['answer']):
            data = json.loads(results['answer'])
        if "isQuestion" in data and "response" in data:
            self.chat_history.append([HumanMessage(content=final_question), data["isQuestion"],
                                      AIMessage(content=data["response"])])
            self.chat_length = self.chat_length + 1
        else:
            return data
        return data["response"]
