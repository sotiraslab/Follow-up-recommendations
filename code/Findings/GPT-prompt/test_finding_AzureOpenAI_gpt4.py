import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables (set OPENAI_API_KEY and OPENAI_API_BASE in .env)
load_dotenv()

# Configure Azure OpenAI Service API
OPENAI_API_TYPE = "azure"
OPENAI_API_VERSION = "2023-08-01-preview"
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = AzureChatOpenAI(
    deployment_name="gpt4",
    model_name="gpt-4",
    temperature=0,
    openai_api_version="2023-08-01-preview"
    )

#load impression section one by one
test_finding_filepath = './Finding/Test/test.txt'

#output predictions
output_file_path = './Finding_AzureOpenAI/predictions_gpt_4.txt'

i = 0
gts = []; texts = []; predictions_gpt_4_turbo = []
with open(test_finding_filepath, 'r') as file:
    for line in file:
        
        
        gt, text = line.split('\t')
        gts.append(gt)
        texts.append(text)

        content = "I will provide you with some text, which belongs to the finding section of a radiology report. Can you tell me whether it contains a recommendation for a follow-up? Please respond with follow-up or no follow-up. The text is the following: "  + text
        message = HumanMessage(content=content)
        
        try:
            prediction = llm([message]).content
        except Exception as e:
            print(f"Error occurred: {e}")
            prediction = text
        
       

        print(prediction)
        predictions_gpt_4_turbo.append(prediction)


with open(output_file_path, 'w') as file:
    for line in predictions_gpt_4_turbo:
        file.write(line + '\n')  # Write each item from the list with a newline character

print(f'Text list saved to {output_file_path}')