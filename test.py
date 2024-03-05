from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

import os
import openai
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai.api_key = "sk-"
persist_directory = os.environ.get('PERSIST_DIRECTORY')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
docsearch = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

def gen_prompt(docs, query) -> str:
    return f"""To answer the question please only use the Context given, nothing else. Using vietnamese to answer
Question: {query}
Context: {[doc.page_content for doc in docs]}
Answer:
"""

def prompt(query):
     docs = docsearch.similarity_search(query, k=4) # tìm kiếm 4 văn bản tương tự 
     prompt = gen_prompt(docs, query)
     return prompt


def stream(input_text):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
            {"role": "system", "content": "You're an assistant."},
            {"role": "user", "content": f"{prompt(input_text)}"},
        ], stream=True, temperature=0)
        print("Answer: ")
        for line in completion:
            # print(line, end='', flush=True)
            if 'content' in line['choices'][0]['delta']:
                # yield line['choices'][0]['delta']['content']  #tương tự return, khác cái là trả về hết vòng lặp thì mới thoát khỏi hàm
                print(line['choices'][0]['delta']['content'],end='', flush=True)

stream("Chương trình đào tạo công nghệ thông tin bao gồm bao nhiêu tín chỉ ?")  
print("\n")
# if __name__ == '__main__':
#     app.run(debug=True)

