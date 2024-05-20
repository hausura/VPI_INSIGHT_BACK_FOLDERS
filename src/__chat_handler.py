import os

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from backend.src.core.database_connection import DatabasePlugin

from semantic_kernel.utils.settings import openai_settings_from_dot_env

class ChatHandler:
    def __init__(self) -> None:
        self.kernel = sk.Kernel()

        api_key, org_id = openai_settings_from_dot_env()
        self.kernel.add_service(
            sk_oai.OpenAIChatCompletion(
                service_id="chat",
                ai_model_id="gpt-3.5-turbo-1106",
                api_key=api_key,
            ),
        )
        self.detection_plugin = self.kernel.add_plugin(parent_directory=os.path.dirname(os.path.realpath(__file__))+"/plugins/", plugin_name="IntentDetectionPlugin")
        self.detection_function = self.detection_plugin["QuestionDetection"]
        database_plugin_object = DatabasePlugin(kernel=self.kernel)
        self.database_plugin = self.kernel.add_plugin(database_plugin_object, "DatabasePlugin")
        self.database_query_function = self.database_plugin["QueryForKnowledgeGraphContent"]
    
    async def handle_chat(self, message: str, chat_history_json: str = None, user_message_included: bool = False) -> tuple[str, str, str]:
        """
        Xử lý tin nhắn tới chatbot.
        
        Args:
            message (str): Tin nhắn tới từ người dùng
            chat_history_json (str): Chuỗi JSON chứa lịch sử chat
            user_message_included (bool): Biến chứa thông tin rằng tin nhắn cuối của người dùng đã được thêm vào lịch sử chat hay chưa
        Returns:
            reply (str): Trả lời từ chatbot
            chat_history (str): JSON chứa lịch sử chat
            data (str): JSON chưa thông tin thêm
        """
        
        if chat_history_json:
            chat_history = ChatHistory.restore_chat_history(chat_history_json)
        else:
            chat_history = ChatHistory()
            system_message = """
            # You are a chat bot. You have one goal: figure out what people need.
            # Once you have the answer I am looking for, 
            # you will return a full answer to me as soon as possible.
            # If the planner provides information, you job is to make that into an answer. 
            """
            chat_history.add_system_message(system_message)
            chat_history.add_user_message("Hi there, who are you?")
            chat_history.add_assistant_message("I am a chat bot. I'm trying to figure out what people need.")
        if not user_message_included:
            chat_history.add_user_message(message)

        intent = await self.kernel.invoke(self.detection_function, KernelArguments(input=chat_history.serialize()))
        if str(intent).strip() == "NO QUESTION":
            reply = "If you have anything, feel free to ask."
            data = None
        else:
            #dummy reply
            data = (await self.kernel.invoke(self.database_query_function, KernelArguments(question = intent))).value
            reply = data["answer"]
        chat_history.add_assistant_message(reply)
        return reply, chat_history.serialize(), str(data)
