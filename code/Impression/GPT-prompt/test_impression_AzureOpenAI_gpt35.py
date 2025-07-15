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
    deployment_name="gpt35",
    model_name="gpt-35-turbo",
    temperature=0,
    openai_api_version="2023-08-01-preview"
    )

#load impression section one by one
test_impression_filepath = './Impression/Test/test.txt'

#output predictions
output_file_path = './Impression_AzureOpenAI/predictions_gpt_3p5_turbo.txt'

i = 0
gts = []; texts = []; predictions_gpt_3p5_turbo = []
with open(test_impression_filepath, 'r') as file:
    for line in file:        

        gt, text = line.split('\t')
        gts.append(gt)
        texts.append(text)

        content = "I will provide you with some text, which belongs to the impression section of a radiology report . Can you tell me whether it contains a recommendation for a follow-up . Please respond with follow-up or no follow-up . The text is the following: "  + text
        message = HumanMessage(content=content)

        try:
            prediction = llm([message]).content
        except Exception as e:
            print(f"Error occurred: {e}")
            prediction = text

        print(prediction)
        predictions_gpt_3p5_turbo.append(prediction)


with open(output_file_path, 'w') as file:
    for line in predictions_gpt_3p5_turbo:
        file.write(line + '\n')  # Write each item from the list with a newline character

print(f'(Finding) Text list saved to {output_file_path}')