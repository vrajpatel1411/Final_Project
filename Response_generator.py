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
from langchain.chains import create_history_aware_retriever

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
        # get_entity = self.ner.predict_entities()
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



    def handle_product_information_search(self, user_query):
        print("Handle Product Information Search Handler")
        final_question = user_query
        prompt_template1 = """
                        you are a key part in chatbot working chain. your work is to take chat history and user input in a mind and 
                        build a context aware sentence of users questions or answer in a way that represent the whole chat.
                        in short you have to make full sentence out of short answers or questions. 

                        for example just for reference 
                        lets say in chat bot history conversation is going on between user and aimessage about a phone 
                        and aimessage asks for more details like which color you want in samsung s20
                        if user input is just black or i need black color
                        so you can genereate a short answers or questions based on the conversation going on and user input
                         like i need black color for samsung s20
                        you output should be like this  "i need black color for samsung s20" 

                        and another things you have to keep in mind like in chat history is conversation is between user and aimessage about a phone
                        name samsung s20 and then if user input comes like what is a price
                        so you have to connect not only to last question but also consider previous conversation from the start
                        so for this example your output should be like this "what is a price for samsung s20?"

                        you have to look at the conversation going on between user and aimessage before creating a context aware sentence
                        like what is it about a perticular product or any information and the you have to try to connect that thing in user input 
                        so retriver can find the best answer.
                        
                        one other thing in chat history please give more priority to the last couple of response from the aimessage and user.
                        sometime in conversations user might change the whole context of chat so keep that in mind.
                        for example sometimes first user is asking about any phone suddenly user might ask about the 
                        laptop in this way context is changed.
                        sometimes user might just answering a question which was asked by aimessage so you just have phrase it well
                        your work is to make best context aware sentence while doing this dont try to change the whole conversation.

                        user input: {input}
                        Chat History: {chat_history}

                                    """

        qa_prompt1 = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=prompt_template1,
        )
        simple_chain = LLMChain(llm=self.llm, prompt=qa_prompt1)
        results = simple_chain.invoke({"input": user_query, "chat_history": self.chat_history})
        print(">>>>>>>>>>>>>>>>>>>newwwwwww", results['text'])
        final_new_question = results["text"]

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
        results = retrieval_chain.invoke({"input": final_new_question, "chat_history": self.chat_history})


        data = None
        if (results['answer']):
            data = json.loads(results['answer'])
            print(data)
        if "answer" in data:
            self.chat_history.append([HumanMessage(content=final_new_question), data['answer']["isQuestion"],
                                      AIMessage(content=data['answer']["response"])])
            self.chat_length = self.chat_length + 1
            return data['answer']['response']
        elif "isQuestion" in data and "response" in data:
            self.chat_history.append([HumanMessage(content=final_new_question), data["isQuestion"],
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

    def handle_search(self, user_query):
        print("Handling search request")
        labels = ["category", "range", "brand", "model name"]
        # entities = self.ner.predict_entities(user_query, labels)
        temp = dict()
        final_question = user_query

        prompt_template1 = """
                        you are a key part in chatbot working chain. your work is to take chat history and user input in a mind and 
                        build a context aware sentence of users questions or answer in a way that represent the whole chat.
                        in short you have to make full sentence out of short answers or questions. 

                        for example just for reference 
                        lets say in chat bot history conversation is going on between user and aimessage about a phone 
                        and aimessage asks for more details like which color you want in samsung s20
                        if user input is just black or i need black color
                        so you can genereate a short answers or questions based on the conversation going on and user input
                         like i need black color for samsung s20
                        you output should be like this  "i need black color for samsung s20" 

                        and another things you have to keep in mind like in chat history is conversation is between user and aimessage about a phone
                        name samsung s20 and then if user input comes like what is a price
                        so you have to connect not only to last question but also consider previous conversation from the start
                        so for this example your output should be like this "what is a price for samsung s20?"

                        you have to look at the conversation going on between user and aimessage before creating a context aware sentence
                        like what is it about a perticular product or any information and the you have to try to connect that thing in user input 
                        so retriver can find the best answer.
                        
                        one other thing in chat history please give more priority to the last couple of response from the aimessage and user.
                        sometime in conversations user might change the whole context of chat so keep that in mind.
                        for example sometimes first user is asking about any phone suddenly user might ask about the 
                        laptop in this way context is changed.
                        sometimes user might just answering a question which was asked by aimessage so you just have phrase it well
                        your work is to make best context aware sentence while doing this dont try to change the whole conversation.

                        user input: {input}
                        Chat History: {chat_history}

                                    """
        qa_prompt1 = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=prompt_template1,
        )
        simple_chain = LLMChain(llm=self.llm, prompt=qa_prompt1)
        results = simple_chain.invoke({"input": user_query, "chat_history": self.chat_history})
        print(">>>>>>>>>>>>>>>>>>>", results['text'])
        final_new_question = results["text"]

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
        results = retrieval_chain.invoke({"input": final_new_question, "chat_history": self.chat_history})
        # print(results)
        data=None
        if(results['answer']):
            data = json.loads(results['answer'])
        if "isQuestion" in data and "response" in data:
            self.chat_history.append([HumanMessage(content=final_new_question),data["isQuestion"], AIMessage(content=["response"])])
            self.chat_length = self.chat_length + 1
        else:
            return data
        return data["response"]

    def handleDirectQuestion(self, user_query):
        print("Handle DirectQuestion")

        prompt_template1 = """
                        you are a key part in chatbot working chain. your work is to take chat history and user input in a mind and 
                        build a context aware sentence of users questions or answer in a way that represent the whole chat.
                        in short you have to make full sentence out of short answers or questions. 

                        for example just for reference 
                        lets say in chat bot history conversation is going on between user and aimessage about a phone 
                        and aimessage asks for more details like which color you want in samsung s20
                        if user input is just black or i need black color
                        so you can genereate a short answers or questions based on the conversation going on and user input
                         like i need black color for samsung s20
                        you output should be like this  "i need black color for samsung s20" 

                        and another things you have to keep in mind like in chat history is conversation is between user and aimessage about a phone
                        name samsung s20 and then if user input comes like what is a price
                        so you have to connect not only to last question but also consider previous conversation from the start
                        so for this example your output should be like this "what is a price for samsung s20?"

                        you have to look at the conversation going on between user and aimessage before creating a context aware sentence
                        like what is it about a perticular product or any information and the you have to try to connect that thing in user input 
                        so retriver can find the best answer.
                        
                        one other thing in chat history please give more priority to the last couple of response from the aimessage and user.
                        sometime in conversations user might change the whole context of chat so keep that in mind.
                        for example sometimes first user is asking about any phone suddenly user might ask about the 
                        laptop in this way context is changed.
                        sometimes user might just answering a question which was asked by aimessage so you just have phrase it well
                        your work is to make best context aware sentence while doing this dont try to change the whole conversation.

                        user input: {input}
                        Chat History: {chat_history}

                                    """

        qa_prompt1 = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=prompt_template1,
        )
        simple_chain = LLMChain(llm=self.llm, prompt=qa_prompt1)
        results = simple_chain.invoke({"input": user_query, "chat_history": self.chat_history})
        print(">>>>>>>>>>>>>>>>>>>", results['text'])
        final_new_question=results["text"]

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
        results = retrieval_chain.invoke({"input": final_new_question, "chat_history": self.chat_history})

        print("contextttttttt",results['context'])

        # print(results)
        data = None
        if (results['answer']):
            data = json.loads(results['answer'])
        if "isQuestion" in data and "response" in data:
            self.chat_history.append([HumanMessage(content=final_new_question), data["isQuestion"],
                                      AIMessage(content=data["response"])])
            self.chat_length = self.chat_length + 1
        else:
            return data
        return data["response"]
