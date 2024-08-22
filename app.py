import os
import gradio as gr
from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.agents.concrete.SimpleConversationAgent import SimpleConversationAgent
from swarmauri.standard.conversations.concrete.Conversation import Conversation

# Fetch the API key from environment variables or define it directly (Not recommended for production)
API_KEY = os.getenv('GROQ_API_KEY')

# Initialize the GroqModel
llm = GroqModel(api_key=API_KEY)

# Create a SimpleConversationAgent with the GroqModel
agent = SimpleConversationAgent(llm=llm, conversation=Conversation())

# Define the function to be executed for the gradio interface
def converse(input_text, history):
    result = agent.exec(input_text)
    return result

demo = gr.ChatInterface(
    fn=converse,
    examples=["Hello!"],
    title="Ask me anything!",
    multimodal=False,
)

if __name__ == "__main__":
    demo.launch()