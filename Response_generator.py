import json

from langchain.chains.llm import LLMChain

from langchain.memory import ConversationBufferMemory

import Bert
import NER
import Ensemble_retriever
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain import hub

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")


class ResponseGenerator:

    def __init__(self):
        self.llm1 = ChatOpenAI(temperature=0.5, openai_api_key="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")
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key="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")
        self.intent_classifier = Bert.BertClassification()
        self.ner = NER.NamedEntityRecognizer()
        self.ensemble = Ensemble_retriever.prepare_dataset()
        self.fixed_entities = dict()
        self.chat_history = []
        self.chat_length = 0
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            max_len=500,
            return_messages=True,
        )

    def getResponse(self, user_query):

        intents = self.intent_classifier.get_prediction(user_query)
        label = intents[0]['label']
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
        prompt_template1 = """
        you're part of a chatbot system where your role is to understand the context of the
        conversation between the user and the chatbot, and then create a new question based on current user-query and the chat history.

        Strictly you should make a question not a sentence or statement.

        Here's a breakdown with examples

        Understanding Context: You analyze the conversation history to grasp the topic being discussed, such as a
        specific product like the Samsung S20 smartphone. You consider the most recent exchanges to prioritize
        relevance and continuity.

Crafting Context-Aware Sentences: If the previous discussion revolved around features of a specific product,
ensure that responses pertain to that product. For instance, if the user inquired about the features of the Motorola
ThinkPhone, subsequent responses should provide details about the Motorola ThinkPhone, maintaining consistency and
relevance.

Maintaining Conversation Flow: Adapt responses to align with the ongoing discussion, especially when transitioning
between different products or topics. If the user shifts focus from one product to another, ensure that responses
reflect the updated context.

Prioritizing Recent Exchanges: While considering the entire conversation, prioritize the most recent exchanges to
ensure timely and relevant responses. This approach helps maintain coherence and addresses the user's current needs
effectively.

Ensuring Accuracy and Relevance: Strive to provide accurate and relevant information based on the ongoing
conversation. Avoid confusion by strictly providing details related to the product the user asked about, preventing
mix-ups between different products.

Context Switching: Sometime the last conversation was about the mobile phones and now user is asking for laptop or either vice versa. So remember to switch such context.

By adhering to these guidelines, you can enhance the chatbot's ability to provide context-aware responses and improve
the overall user experience. Strictly provide details related to the product user asked. Don't confused among
multiple products. Strictly provide details related to the product user asked.
        Don't confused among multiple products.

         When responding to the user's current input, ensure that if the last conversation explicitly asked the user
         for options, prioritize that response as the primary context. This ensures that the current interaction
         remains focused on addressing the options provided, and allows for a seamless continuation of the
         conversation. Additionally, consider edge cases such as scenarios where the user's current input may not
         directly relate to the options provided in the previous response. In such cases, it's essential to
         gracefully acknowledge the divergence while still maintaining coherence in the conversation flow.

         Note : Chat_history is python list. So last conversation response is added last in the list. Last
         conversational response should get higher priority.

         user input: {input}
         Chat History: {chat_history}
         response:

                                    """
        qa_prompt1 = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=prompt_template1,
        )
        simple_chain = LLMChain(llm=self.llm1, prompt=qa_prompt1)
        results = simple_chain.invoke({"input": user_query, "chat_history": self.chat_history})

        final_new_question = "The original question by user: " + user_query + ". Question generated by combining both chat history and current question : " + \
                             results['text'] + "."

        prompt_template = """As an expert in electronic devices, your role is to offer insightful responses based on 
        user queries, chat history, and context. If there are multiple matching products, present all options. 
        Utilize past conversations and prompt for additional information if necessary. Responses should be concise, 
        accurate, and presented in proper string format, summarizing product details including features, specifications, 
        pricing, and suitability for different computing needs.Give direct response if any specific product name is provided. Respectfully assist users while refraining from 
        misinformation. Strictly provide details related to the product user asked. Don't confused among multiple products.
        
         Note : Chat_history is python list. So last conversation response is added last in the list. Last 
         conversational response should get higher priority.
        
                        User Input : {input} 
                        Chat History : {chat_history} 
                        Retrieved Document: {context}
                        response : 
    
    Below are the example prompt . Take it as an reference.                    
                        ---

                            Prompt for iPhone options:
                            User: [User's query about iPhone options]
                            AI: [Based on the user's query, provide a response suggesting iPhone models with diverse specifications and prices. Avoid repeating the same options in consecutive responses. Offer at least three different iPhone models in each response, ensuring variation in features, prices, and colors.]
                            
                            Prompt for gaming laptops:
                            User: [User's query about gaming laptops]
                            AI: [Based on the user's query, provide a response suggesting gaming laptops with diverse specifications and prices. Ensure each response includes at least three different gaming laptop models from various brands, with variations in processor, graphics card, display, storage, RAM, and price range.]
                        ----
                        
                        Also, make sure when your provide multiple options to user it should be formated like this:
                        
                        ----
                        Here are the few option to your question:
                            Option 1 : product 1 - Description
                            Option 2 : product 2 - Description
                            Option 3 : product 3 - Description
                            and so on...
                            
                        ----
                            
                                             """
        qa_prompt = PromptTemplate(
            input_variables=["input", "chat_history", "context"],
            template=prompt_template,
        )

        combine_docs_chain = create_stuff_documents_chain(
            self.llm, qa_prompt
        )
        try:
            retrieval_chain = create_retrieval_chain(self.ensemble.build_ensemble_retriever(), combine_docs_chain)
            results = retrieval_chain.invoke({"input": final_new_question, "chat_history": self.chat_history})
        except:
            return self.handleDirectQuestion(final_new_question)
        self.chat_history.append([HumanMessage(content=final_new_question), AIMessage(content=results['answer'])])
        self.chat_length = len(self.chat_history)

        if self.chat_length > 5:
            self.chat_history = self.chat_history[1:]
        print(self.chat_history)
        return results['answer']

    # Your code to search for product information

    def handle_unknown_request(self, input):
        return "Unknown Request Handler"

    # Your code to handle unknown requests

    def handle_welcome(self, input):
        prompt_template = """

               You're an experienced electronic expert who's been working in sales for two decades. Your goal is to 
               greet users warmly, introduce yourself briefly, and respond to their queries in a friendly and concise 
               manner, considering past conversations. If there's any uncertainty about the product, 
               ask for clarification. Utilize your extensive experience to provide helpful responses, avoiding any 
               negative feedback. 
                
                Input Question : {input}
                Chat History:{chat_history}
                Response:
                """
        qa_prompt = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=prompt_template,
        )
        simple_chain = LLMChain(llm=self.llm, prompt=qa_prompt)
        results = simple_chain.invoke({"input": input, "chat_history": self.chat_history})
        self.chat_history.append([HumanMessage(content=input), False, results["text"]])
        self.chat_length = self.chat_length + 1
        if self.chat_length > 5:
            self.chat_history = self.chat_history[1:]
        return results['text']

    def handle_search(self, user_query):
        print("Handling search request")

        prompt_template1 = """
        
        you're part of a chatbot system where your role is to understand the context of the 
        conversation between the user and the chatbot, and then create a new question based on current user-query and the chat history. 
        
        Strictly you should make a question not a sentence or statement.
        
       

       Understanding Context: You analyze the conversation history to grasp the topic being discussed, such as a 
       specific product like the Samsung S20 smartphone. You consider the most recent exchanges to prioritize 
       relevance and continuity.

        Crafting Context-Aware Sentences: If the previous discussion revolved around features of a specific product, 
        ensure that responses pertain to that product. For instance, if the user inquired about the features of the Motorola 
        ThinkPhone, subsequent responses should provide details about the Motorola ThinkPhone, maintaining consistency and 
        relevance.
        
        Maintaining Conversation Flow: Adapt responses to align with the ongoing discussion, especially when transitioning 
        between different products or topics. If the user shifts focus from one product to another, ensure that responses 
        reflect the updated context.
        
        Prioritizing Recent Exchanges: While considering the entire conversation, prioritize the most recent exchanges to 
        ensure timely and relevant responses. This approach helps maintain coherence and addresses the user's current needs 
        effectively.
        
        Ensuring Accuracy and Relevance: Strive to provide accurate and relevant information based on the ongoing 
        conversation. Avoid confusion by strictly providing details related to the product the user asked about, preventing 
        mix-ups between different products.
        
        Context Switching: Sometime the last conversation was about the mobile phones and now user is asking for laptop or either vice versa. So remember to switch such context.
        By adhering to these guidelines, you can enhance the chatbot's ability to provide context-aware responses and improve 
        the overall user experience. Strictly provide details related to the product user asked. Don't confused among 
        multiple products.
         
         When responding to the user's current input, ensure that if the last conversation explicitly asked the user 
         for options, prioritize that response as the primary context. This ensures that the current interaction 
         remains focused on addressing the options provided, and allows for a seamless continuation of the 
         conversation. Additionally, consider edge cases such as scenarios where the user's current input may not 
         directly relate to the options provided in the previous response. In such cases, it's essential to 
         gracefully acknowledge the divergence while still maintaining coherence in the conversation flow. 
         
         Note : Chat_history is python list. So last conversation response is added last in the list. Last conversational 
         response should get higher priority..
         
         user input: {input} 
         Chat History: {chat_history}
         response:

                                    """
        qa_prompt1 = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=prompt_template1,
        )
        simple_chain = LLMChain(llm=self.llm1, prompt=qa_prompt1)
        results = simple_chain.invoke({"input": user_query, "chat_history": self.chat_history})

        final_new_question = "The original question by user: " + user_query + (". Question generated by combining both "
                                                                               "chat history and current question : "
                                                                               "") + \
                             results['text'] + "."

        prompt_template = """As an expert in electronic devices, your role is to offer insightful responses based on 
        user queries, chat history, and context. If there are multiple matching products, present all options. 
        Utilize past conversations and prompt for additional information if necessary. Responses should be concise, 
        accurate, and presented in proper string format, summarizing product details including features, specifications, 
        pricing, and suitability for different computing needs. Give direct response if any specific product name is provided. Respectfully assist users while refraining from 
        misinformation. Strictly provide details related to the product user asked. Don't confused among multiple products.
        
         Note : Chat_history is python list. So last conversation response is added last in the list. Last 
         conversational response should get higher priority.
        
                        User Input : {input} 
                        Chat History : {chat_history} 
                        Retrieved Document: {context}
                        response : 
            
             Below are the example prompt . Take it as an reference.                    
                        ---

                            Prompt for iPhone options:
                            User: [User's query about iPhone options]
                            AI: [Based on the user's query, provide a response suggesting iPhone models with diverse specifications and prices. Avoid repeating the same options in consecutive responses. Offer at least three different iPhone models in each response, ensuring variation in features, prices, and colors.]
                            
                            Prompt for gaming laptops:
                            User: [User's query about gaming laptops]
                            AI: [Based on the user's query, provide a response suggesting gaming laptops with diverse specifications and prices. Ensure each response includes at least three different gaming laptop models from various brands, with variations in processor, graphics card, display, storage, RAM, and price range.]
                        ----
                        
                        Also, make sure when your provide multiple options to user it should be formated like this:
                        
                        ----
                        Here are the few option to your question:
                            Option 1 : product 1 - Description
                            Option 2 : product 2 - Description
                            Option 3 : product 3 - Description
                            and so on...
                        ----
                        """
        qa_prompt = PromptTemplate(
            input_variables=["input", "chat_history", "context"],
            template=prompt_template,
        )
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, qa_prompt
        )
        try:
            retrieval_chain = create_retrieval_chain(self.ensemble.build_ensemble_retriever(), combine_docs_chain)
            results = retrieval_chain.invoke({"input": user_query, "chat_history": self.chat_history})

        except:
            return self.handleDirectQuestion(user_query)
        print(results)

        print("==============================================")

        self.chat_history.append([HumanMessage(content=user_query), AIMessage(content=results['answer'])])
        self.chat_length = len(self.chat_history)
        print(self.chat_history)
        if self.chat_length > 5:
            self.chat_history = self.chat_history[1:]

        return results['answer']

    def handleDirectQuestion(self, user_query):
        print("Handle DirectQuestion")

        prompt_template2 = """As an expert in electronic devices, your role is to offer insightful responses based on 
        user queries, chat history, and context. If there are multiple matching products, present all options. 
        Utilize past conversations and prompt for additional information if necessary. Responses should be concise, 
        accurate, and presented in proper string format, summarizing product details including features, specifications, 
        pricing, and suitability for different computing needs. Give direct response if any specific product name is provided. Respectfully assist users while refraining from 
        misinformation. Strictly provide details related to the product user asked. Don't confused among multiple products.
        
         Note : Chat_history is python list. So last conversation response is added last in the list. Last 
         conversational response should get higher priority.
         
                        User Input : {input} 
                        Chat History : {chat_history} 
                        Retrieved Document: {context}
                        response : 
        
         Below are the example prompt . Take it as an reference.                    
                        ---

                            Prompt for iPhone options:
                            User: [User's query about iPhone options]
                            AI: [Based on the user's query, provide a response suggesting iPhone models with diverse specifications and prices. Avoid repeating the same options in consecutive responses. Offer at least three different iPhone models in each response, ensuring variation in features, prices, and colors.]
                            
                            Prompt for gaming laptops:
                            User: [User's query about gaming laptops]
                            AI: [Based on the user's query, provide a response suggesting gaming laptops with diverse specifications and prices. Ensure each response includes at least three different gaming laptop models from various brands, with variations in processor, graphics card, display, storage, RAM, and price range.]
                        ----
                        
                        Also, make sure when your provide multiple options to user it should be formated like this:
                        
                        ----
                        Here are the few option to your question:
                            Option 1 : product 1 - Description
                            Option 2 : product 2 - Description
                            Option 3 : product 3 - Description
                            and so on...
                        ----
                        """
        qa_prompt2 = PromptTemplate(
            input_variables=["input", "chat_history", "context"],
            template=prompt_template2,
        )
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, qa_prompt2
        )
        results = None
        try:
            retrieval_chain = create_retrieval_chain(self.ensemble.build_ensemble_retriever(), combine_docs_chain)
            results = retrieval_chain.invoke({"input": user_query, "chat_history": self.chat_history})
        except:
            self.chat_history.append([HumanMessage(content=user_query),
                                      AIMessage(content="Sorry, Couldn't get it. Can you please, try again?")])
            self.chat_length = len(self.chat_history)
            if self.chat_length > 5:
                self.chat_history = self.chat_history[1:]
            return "Sorry, Couldn't get it. Can you please, try again?"
        self.chat_history.append([HumanMessage(content=user_query), AIMessage(content=results['answer'])])
        self.chat_length = len(self.chat_history)
        print(self.chat_history)
        if self.chat_length > 5:
            self.chat_history = self.chat_history[1:]
        return results['answer']
