import ast
import json

from langchain.chains.llm import LLMChain

from langchain.memory import ConversationBufferMemory

import Bert
import NER
import Ensemble_retriever
import sqlDB
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain import hub

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")


prompt_for_new_question="""

Imagine you're a super-powered conversation assistant! I'm here to assist you in the noble quest of understanding 
user queries within a chatbot conversation.  I'll analyze the chat history to grasp the context, especially focusing 
on the latest user input.  My job is to craft a concise question that effectively captures the current user's intent 
and the overall conversation flow.

Here's your secret playbook:

Be a Context Master: I'll meticulously analyze the chat_history like a digital Sherlock Holmes, piecing together the 
conversation's main topic and direction.

Stay on Topic: If the conversation revolved around a specific product, like the Motorola ThinkPhone, my questions 
will stay on track, focusing on that product to maintain consistency. No jumping to different phones unless the user 
clearly shifts the focus.

Flow Like a Pro: As the conversation changes topics (phones to laptops, for instance), I'll smoothly adjust my 
questions to match the new direction. I'll be like a conversational chameleon, adapting to keep things relevant.

Focus on the Now: The most recent user input (user_input) will receive my highest attention. This ensures the 
generated question reflects their current needs (Prioritize the Latest). 

Accuracy is Key: My questions will be relevant and based on the conversation. Don't get confused and mix up product 
details (Accuracy is Key).

Crafting the Perfect Query: Based on my analysis, I'll formulate a clear and concise question that accurately 
embodies the user's intent within the conversation's context.

By following these guidelines, you'll become a master of conversation flow, making the chatbot experience smooth and 
enjoyable for everyone.


user_input: {input} ,  This is the text the user just typed. chat_history: {chat_history} ,  This is a list that 
stores all the previous messages in the conversation, with the most recent message being the last item in the list. 
Your Response (response):

 Below are the example prompt . Take it as an reference to understand your job not to create a response.
    << EXAMPLE 1 >>
        chat_history=[[HumanMessage(content=Laptop), AIMessage(content=Here are some laptop options based on your query:
            
            Option 1:  Device 1 (14 inch display)
                    Information about the device 1
            Option 2: Device 2 (15.6 inch display)
                    Information about the device 2  
            Option 3: Device 3 (15.6 inch anti-glare display)
                    Information about the device 3  )]]
        
        user_input: Provide more information for the 3rd option
        Your Response: Can you provide more details about the Device 3? 

        


    << EXAMPLE 2 >>
     
            chat_history=[[HumanMessage(content=Laptop), AIMessage(content=Here are some laptop options based on your query:
            
            Option 1:  Device 1 (14 inch display)
                    Information about the device 1
            Option 2: Device 2 (15.6 inch display)
                    Information about the device 2  
            Option 3: Device 3 (15.6 inch anti-glare display)
                    Information about the device 3  )]]
             
            user_input:  What's the price for Device 3?
            Your Response:  Can you tell me the price range for Device 3?
            
            
"""


prompt_for_product= """In this exciting role, you'll be a master of electronic devices, guiding users with your 
        knowledge and expertise.

        Your Mission:

        Understand User Needs: Analyze user queries, chat history, and additional context to grasp what they're 
        looking for. 
        
        Offer Smart Options: If multiple products fit the user request, present a variety of choices 
        with diverse features, specifications, and prices. 
        
        Leverage Past Conversations: Utilize past interactions to tailor your responses and avoid suggesting the same 
        options repeatedly. Ask for Clarification (when needed): If necessary, politely prompt the user for more 
        information to provide the most relevant suggestions. 
        
        Provide Concise & Accurate Info: Craft clear and informative summaries of product details, including 
        features, specifications, pricing, and suitability for different needs. 
        
        Be a Product Whisperer: When a specific product name is mentioned, focus your response entirely on that 
        device. Maintain Respectful & Honest Communication: Always be helpful and avoid providing misleading 
        information. 
        
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

class ResponseGenerator:

    def __init__(self):
        self.llm1 = ChatOpenAI(temperature=0.5, openai_api_key="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")
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key="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")
        self.intent_classifier = Bert.BertClassification()
        self.ner = NER.NamedEntityRecognizer()
        self.db = sqlDB.DatabaseConnection()
        self.ensemble = Ensemble_retriever.prepare_dataset()
        self.fixed_entities = dict()
        self.chat_history = []
        self.product_history = []
        self.order_history =[]
        self.chat_length = 0
        self.order_chat_length = 0
        self.product_chat_length = 0
        self.email = ''
        self.order_id = ''
        self.isTrackOrderFollowup = False
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            max_len=500,
            return_messages=True,
        )

    def create_new_question(self,user_query,chat_history):
        # prompt_template1 = """
        #                            you're part of a chatbot system where your role is to understand the context of the
        #                            conversation between the user and the chatbot, and then create a new question based on current user-query and the chat history.
        #
        #                            Strictly you should make a question not a sentence or statement.
        #
        #                            Here's a breakdown with examples
        #
        #                            Understanding Context: You analyze the conversation history to grasp the topic being discussed, such as a
        #                            specific product like the Samsung S20 smartphone. You consider the most recent exchanges to prioritize
        #                            relevance and continuity.
        #
        #                    Crafting Context-Aware Sentences: If the previous discussion revolved around features of a specific product,
        #                    ensure that responses pertain to that product. For instance, if the user inquired about the features of the Motorola
        #                    ThinkPhone, subsequent responses should provide details about the Motorola ThinkPhone, maintaining consistency and
        #                    relevance.
        #
        #                    Maintaining Conversation Flow: Adapt responses to align with the ongoing discussion, especially when transitioning
        #                    between different products or topics. If the user shifts focus from one product to another, ensure that responses
        #                    reflect the updated context.
        #
        #                    Prioritizing Recent Exchanges: While considering the entire conversation, prioritize the most recent exchanges to
        #                    ensure timely and relevant responses. This approach helps maintain coherence and addresses the user's current needs
        #                    effectively.
        #
        #                    Ensuring Accuracy and Relevance: Strive to provide accurate and relevant information based on the ongoing
        #                    conversation. Avoid confusion by strictly providing details related to the product the user asked about, preventing
        #                    mix-ups between different products.
        #
        #                    Context Switching: Sometime the last conversation was about the mobile phones and now user is asking for laptop or either vice versa. So remember to switch such context.
        #
        #                    By adhering to these guidelines, you can enhance the chatbot's ability to provide context-aware responses and improve
        #                    the overall user experience. Strictly provide details related to the product user asked. Don't confused among
        #                    multiple products. Strictly provide details related to the product user asked.
        #                            Don't confused among multiple products.
        #
        #                             When responding to the user's current input, ensure that if the last conversation explicitly asked the user
        #                             for options, prioritize that response as the primary context. This ensures that the current interaction
        #                             remains focused on addressing the options provided, and allows for a seamless continuation of the
        #                             conversation. Additionally, consider edge cases such as scenarios where the user's current input may not
        #                             directly relate to the options provided in the previous response. In such cases, it's essential to
        #                             gracefully acknowledge the divergence while still maintaining coherence in the conversation flow.
        #
        #                             Note : Chat_history is python list. So last conversation response is added last in the list. Last
        #                             conversational response should get higher priority.
        #
        #                             user input: {input}
        #                             Chat History: {chat_history}
        #                             response:
        # """
        qa_prompt1 = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=prompt_for_new_question,
        )
        simple_chain = LLMChain(llm=self.llm1, prompt=qa_prompt1)
        results = simple_chain.invoke({"input": user_query, "chat_history": chat_history})
        final_new_question = "The original question by user: " + user_query + ". Question generated by combining both chat history and current question : " + \
                             results['text'] + "."
        print(results["text"])
        return final_new_question

    def getResponse(self, user_query):
        intents = self.intent_classifier.get_prediction(user_query)
        if self.isTrackOrderFollowup:
            response = self.handle_order_tracking(user_query)
            return response
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
        if(label == 'unknown'):
            results=self.create_new_question(user_query,self.chat_history)
            intents = self.intent_classifier.get_prediction(results)
            label=intents[0]['label']
            if(label == 'unknown'):
                return self.handle_unknown_request(user_query)
            self.handleDirectQuestion(results)
        handler = intent_handlers.get(label, self.handle_unknown_request)
        response = handler(user_query)
        return response

    def handle_refund(self, user_query):
        return "Refund Handler"

    # Your code to handle refund request

    def handle_order_cancellation(self, user_query):
        return "Order Cancellation Handler"

    # Your code to handle order cancellation

    def handle_human_contact(self, user_query):
        return "Human Contact Handler"

    # Your code to contact a human

    def handle_payment_issue(self, user_query):
        return "Payment Issue Handler"

    # Your code to handle payment issues

    def handle_password_recovery(self, user_query):
        return "Password Recovery Handler"

    # Your code to recover password

    def handle_farewell(self, user_query):
        return "Fairwell Handler"

    # Your code to bid farewell

    def handle_order_tracking(self, user_query):
        order_details = ''
        if self.order_id:
            r = self.db.findOne(
                f"SELECT order_status,estimate FROM Orders WHERE order_id = {self.order_id}")

            if r:
                order_details=f"status of order : {r[0]} and estimate time to delivery : {r[1]} days"
            else:
                order_details= "no data found"
        print("Handle order tracking Handler")
        final_new_question = self.create_new_question(user_query,self.order_history)
        entity_names = ["email", "order id"]
        entity = self.ner.predict_entities(final_new_question, entity_names)
        if entity:
            # Filter entries with score greater than 90
            filtered_data = [item for item in entity if item['score'] > 0.9]

            # If there are any entries with score above 90, extract label and text
            if filtered_data:
                label = filtered_data[0]['label']
                text = filtered_data[0]['text']
                print(f"Label: {label}, Text: {text}")
                if label == "email":
                    self.email = text
                if label == "order id":
                    self.order_id = text
            else:
                print("No entries found with score above 90")
        prompt_template = """
        you are a customer service representative you customer's inquiries about the tracking their order
        your job is to develop a follow up questions or answer the customers questions based in provided context
        
        if Order Id is empty then ask customer for order Id 
        and if you have the order id and Retrieved Document then answer the customers questions based on Retrieved Document. 
        
                 Note : Chat_history is python list. So last conversation response is added last in the list. Last 
                 conversational response should get higher priority.

                                User Input : {input} 
                                Chat History : {chat_history} 
                                Retrieved Document: {context}
                                Order Id : {order_id}
                                response : 

        take these examples as reference 
        user: [user ask to order tracking]
        AI : [ask for order id if order id is empty]
        
        you should just return response like
        text : "your response"
        isTrackOrderFollowup : "boolean"
        
        
        in your response isTrackOrderFollowup should True if are asking for more information from user and False if 
        you are just answer the customers questions based on Retrieved Document.
        
        your final response should be in JSON format strictly instead of string with just two key fields "text" and "isTrackOrderFollowup"
                                ----

                                                     """
        qa_prompt = PromptTemplate(
            input_variables=["input", "chat_history", "context","order_id"],
            template=prompt_template,
        )

        simple_chain = LLMChain(llm=self.llm, prompt=qa_prompt)
        results = simple_chain.invoke({"input": user_query, "chat_history": self.chat_history,"order_id":self.order_id,"context":order_details})
        print("???????????????",results["text"])
        text = ''
        if results['text']:
            text = json.loads(results["text"])["text"]
            self.isTrackOrderFollowup = json.loads(results["text"])['isTrackOrderFollowup']

        self.chat_history.append([HumanMessage(content=final_new_question), AIMessage(content=text)])
        self.order_history.append([HumanMessage(content=final_new_question), AIMessage(content=text)])
        self.order_chat_length = len(self.order_history)

        if self.order_chat_length > 5:
            self.order_history = self.order_history[1:]
        print(self.order_history)
        return text



    def handle_product_information_search(self, user_query):
        print("Handle Product Information Search Handler")

        final_new_question=self.create_new_question(user_query,self.product_history)
        # prompt_template = """In this exciting role, you'll be a master of electronic devices, guiding users with your
        # knowledge and expertise.
        #
        # Your Mission:
        #
        # Understand User Needs: Analyze user queries, chat history, and additional context to grasp what they're
        # looking for.
        #
        # Offer Smart Options: If multiple products fit the user request, present a variety of choices
        # with diverse features, specifications, and prices.
        #
        # Leverage Past Conversations: Utilize past interactions to tailor your responses and avoid suggesting the same
        # options repeatedly. Ask for Clarification (when needed): If necessary, politely prompt the user for more
        # information to provide the most relevant suggestions.
        #
        # Provide Concise & Accurate Info: Craft clear and informative summaries of product details, including
        # features, specifications, pricing, and suitability for different needs.
        #
        # Be a Product Whisperer: When a specific product name is mentioned, focus your response entirely on that
        # device. Maintain Respectful & Honest Communication: Always be helpful and avoid providing misleading
        # information.
        #
        # User Input : {input}
        # Chat History : { chat_history}
        # Retrieved Document: {context}
        # response :
        #
        # Below are the example prompt . Take it as an reference.
        #                 ---
        #
        #                     Prompt for iPhone options:
        #                     User: [User's query about iPhone options]
        #                     AI: [Based on the user's query, provide a response suggesting iPhone models with diverse specifications and prices. Avoid repeating the same options in consecutive responses. Offer at least three different iPhone models in each response, ensuring variation in features, prices, and colors.]
        #
        #                     Prompt for gaming laptops:
        #                     User: [User's query about gaming laptops]
        #                     AI: [Based on the user's query, provide a response suggesting gaming laptops with diverse specifications and prices. Ensure each response includes at least three different gaming laptop models from various brands, with variations in processor, graphics card, display, storage, RAM, and price range.]
        #                 ----
        #
        #                 Also, make sure when your provide multiple options to user it should be formated like this:
        #
        #                 ----
        #                 Here are the few option to your question:
        #                     Option 1 : product 1 - Description
        #                     Option 2 : product 2 - Description
        #                     Option 3 : product 3 - Description
        #                     and so on...
        #
        #                 ----
        #
        #                                      """
        qa_prompt = PromptTemplate(
            input_variables=["input", "chat_history", "context"],
            template=prompt_for_product,
        )

        combine_docs_chain = create_stuff_documents_chain(
            self.llm, qa_prompt
        )
        try:
            retrieval_chain = create_retrieval_chain(self.ensemble.build_ensemble_retriever(), combine_docs_chain)
            results = retrieval_chain.invoke({"input": final_new_question, "chat_history": self.product_history})
        except:
            return self.handleDirectQuestion(final_new_question)
        self.chat_history.append([HumanMessage(content=final_new_question), AIMessage(content=results['answer'])])
        self.product_history.append([HumanMessage(content=final_new_question), AIMessage(content=results['answer'])])
        self.product_chat_length = len(self.product_history)

        if self.product_chat_length  > 5:
            self.product_history = self.product_history[1:]
        print(self.product_history)
        return results['answer']

    # Your code to search for product information

    def handle_unknown_request(self, user_query):
        return "Unknown Request Handler"

    # Your code to handle unknown requests

    def handle_welcome(self, user_query):
        prompt_template = """

              Welcome to the Tech Help Desk!

                I'm your friendly tech expert, here to guide you through the world of electronics. I'm happy to 
                answer your questions and help you find the perfect device for your needs.

                How can I assist you today?

                Here's what I can do:

                Understand Your Needs: I'll carefully consider your questions, past conversations, and any additional 
                context to provide the most relevant information.
                
                Offer Clear & Concise Answers: I'll break down complex tech details into easy-to-understand 
                explanations, focusing on features, specifications, pricing, and suitability for your specific needs.

                Multiple Options? No Problem! If there are several products that fit your criteria, I'll present a 
                variety of choices with diverse features and price points.

                Clarification is Key: If something's unclear, I'll politely ask for more information to ensure my 
                recommendations are spot-on. Positive Interactions: My goal is to provide a helpful and informative 
                experience, avoiding any misleading information or negativity.
                
                Input Question : {input}
                Chat History:{chat_history}
                Response:
                
                
                Here's an Example:

User: "What's the best laptop for gaming?" You: "Great question! To recommend the perfect gaming laptop, I'd need to 
know a bit more about your budget and the types of games you usually play. Are you looking for something 
ultra-powerful for the latest AAA titles, or something more balanced for everyday gaming?

"""
        qa_prompt = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=prompt_template,
        )
        simple_chain = LLMChain(llm=self.llm, prompt=qa_prompt)
        results = simple_chain.invoke({"input": user_query, "chat_history": self.chat_history})
        self.chat_history.append([HumanMessage(content=user_query), False, results["text"]])
        self.chat_length = self.chat_length + 1
        if self.chat_length > 5:
            self.chat_history = self.chat_history[1:]
        return results['text']

    def handle_search(self, user_query):
        print("Handling search request")

        final_new_question=self.create_new_question(user_query,self.product_history)
        # prompt_template = """As an expert in electronic devices, your role is to offer insightful responses based on
        # user queries, chat history, and context. If there are multiple matching products, present all options.
        # Utilize past conversations and prompt for additional information if necessary. Responses should be concise,
        # accurate, and presented in proper string format, summarizing product details including features, specifications,
        # pricing, and suitability for different computing needs. Give direct response if any specific product name is provided. Respectfully assist users while refraining from
        # misinformation. Strictly provide details related to the product user asked. Don't confused among multiple products.
        #
        #  Note : Chat_history is python list. So last conversation response is added last in the list. Last
        #  conversational response should get higher priority.
        #
        #                 User Input : {input}
        #                 Chat History : {chat_history}
        #                 Retrieved Document: {context}
        #                 response :
        #
        #      Below are the example prompt . Take it as an reference.
        #                 ---
        #
        #                     Prompt for iPhone options:
        #                     User: [User's query about iPhone options]
        #                     AI: [Based on the user's query, provide a response suggesting iPhone models with diverse specifications and prices. Avoid repeating the same options in consecutive responses. Offer at least three different iPhone models in each response, ensuring variation in features, prices, and colors.]
        #
        #                     Prompt for gaming laptops:
        #                     User: [User's query about gaming laptops]
        #                     AI: [Based on the user's query, provide a response suggesting gaming laptops with diverse specifications and prices. Ensure each response includes at least three different gaming laptop models from various brands, with variations in processor, graphics card, display, storage, RAM, and price range.]
        #                 ----
        #
        #                 Also, make sure when your provide multiple options to user it should be formated like this:
        #
        #                 ----
        #                 Here are the few option to your question:
        #                     Option 1 : product 1 - Description
        #                     Option 2 : product 2 - Description
        #                     Option 3 : product 3 - Description
        #                     and so on...
        #                 ----
        #                 """
        qa_prompt = PromptTemplate(
            input_variables=["input", "chat_history", "context"],
            template=prompt_for_product,
        )
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, qa_prompt
        )
        try:
            retrieval_chain = create_retrieval_chain(self.ensemble.build_ensemble_retriever(), combine_docs_chain)
            results = retrieval_chain.invoke({"input": final_new_question, "chat_history": self.product_history})

        except:
            return self.handleDirectQuestion(user_query)
        print(results)

        print("==============================================")

        self.chat_history.append([HumanMessage(content=final_new_question), AIMessage(content=results['answer'])])
        self.product_history.append([HumanMessage(content=final_new_question), AIMessage(content=results['answer'])])
        self.product_chat_length = len(self.product_history)

        if self.product_chat_length > 5:
            self.product_history = self.product_history[1:]
        print(self.product_history)
        return results['answer']

    def handleDirectQuestion(self, user_query):
        print("Handle DirectQuestion")

        # prompt_template2 = """As an expert in electronic devices, your role is to offer insightful responses based on
        # user queries, chat history, and context. If there are multiple matching products, present all options.
        # Utilize past conversations and prompt for additional information if necessary. Responses should be concise,
        # accurate, and presented in proper string format, summarizing product details including features, specifications,
        # pricing, and suitability for different computing needs. Give direct response if any specific product name is provided. Respectfully assist users while refraining from
        # misinformation. Strictly provide details related to the product user asked. Don't confused among multiple products.
        #
        #  Note : Chat_history is python list. So last conversation response is added last in the list. Last
        #  conversational response should get higher priority.
        #
        #                 User Input : {input}
        #                 Chat History : {chat_history}
        #                 Retrieved Document: {context}
        #                 response :
        #
        #  Below are the example prompt . Take it as an reference.
        #                 ---
        #
        #                     Prompt for iPhone options:
        #                     User: [User's query about iPhone options]
        #                     AI: [Based on the user's query, provide a response suggesting iPhone models with diverse specifications and prices. Avoid repeating the same options in consecutive responses. Offer at least three different iPhone models in each response, ensuring variation in features, prices, and colors.]
        #
        #                     Prompt for gaming laptops:
        #                     User: [User's query about gaming laptops]
        #                     AI: [Based on the user's query, provide a response suggesting gaming laptops with diverse specifications and prices. Ensure each response includes at least three different gaming laptop models from various brands, with variations in processor, graphics card, display, storage, RAM, and price range.]
        #                 ----
        #
        #                 Also, make sure when your provide multiple options to user it should be formated like this:
        #
        #                 ----
        #                 Here are the few option to your question:
        #                     Option 1 : product 1 - Description
        #                     Option 2 : product 2 - Description
        #                     Option 3 : product 3 - Description
        #                     and so on...
        #                 ----
        #                 """
        qa_prompt2 = PromptTemplate(
            input_variables=["input", "chat_history", "context"],
            template=prompt_for_product,
        )
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, qa_prompt2
        )
        results = None
        try:
            retrieval_chain = create_retrieval_chain(self.ensemble.build_ensemble_retriever(), combine_docs_chain)
            results = retrieval_chain.invoke({"input": user_query, "chat_history": self.product_history})
        except:
            self.chat_history.append([HumanMessage(content=user_query),
                                      AIMessage(content="Sorry, Couldn't get it. Can you please, try again?")])
            self.product_chat_length = len(self.product_history)
            if self.product_chat_length > 5:
                self.product_history = self.product_history[1:]
            return "Sorry, Couldn't get it. Can you please, try again?"
        self.chat_history.append([HumanMessage(content=user_query), AIMessage(content=results['answer'])])
        self.product_history.append([HumanMessage(content=user_query), AIMessage(content=results['answer'])])
        self.product_chat_length = len(self.product_history)
        if self.product_chat_length > 5:
            self.product_history = self.product_history[1:]
        print(self.product_history)
        return results['answer']
