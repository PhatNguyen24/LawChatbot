from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp, GPT4All, OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from constants import CHROMA_SETTINGS
import os
import openai

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
n_gpu_layers = os.environ.get('N_GPU_LAYERS')
n_batch = os.environ.get('N_BATCH')

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

def gen_prompt(docs, query) -> str:
    return f"""To answer the question please only use the Context given, nothing else. 
        Using vietnamese to answer question
        Question: {query}
        Context: {[doc.page_content for doc in docs]}
        Answer:
    """
def prompt(query):
     docs = db.similarity_search(query, k=4) # tìm kiếm 4 văn bản tương tự 
     prompt = gen_prompt(docs, query)
     return prompt
def stream(input_text):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
            {"role": "system", "content": "You're an assistant."},
            {"role": "user", "content": f"{prompt(input_text)}"},
        ], stream=True, temperature=0)
        for line in completion:
            # print(line, end='', flush=True)
            if 'content' in line['choices'][0]['delta']:
                # yield line['choices'][0]['delta']['content']  #tương tự return, khác cái là trả về hết vòng lặp thì mới thoát khỏi hàm
                print(line['choices'][0]['delta']['content'],end='', flush=True)

def main():
    template="""Use the following pieces of information to answer the user's question.
    If you dont know the answer just say you know, don't try to make up an answer. Please use Vietnamese to answer the question

    Context:{context}
    Question:{question}

    Only return the helpful answer below and nothing else
    Helpful answer
    """

    qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])
    retriever = db.as_retriever(search_kwargs={'k': 1})   
    callbacks = [StreamingStdOutCallbackHandler()]  # activate/deactivate the streaming StdOut callback for LLMs
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_batch=n_batch,
                callbacks=callbacks,
                verbose=True,  # Verbose is required to pass to the callback manager
                n_ctx=model_n_ctx
            )
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case "ChatGPT":
            exit
        case _default:
            print(f"Model {model_type} not supported!")
            exit

    if model_type != "ChatGPT":
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents= True, 
            chain_type_kwargs={'prompt': qa_prompt}
        )
    # print(qa_prompt)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        if model_type == "ChatGPT":
            stream(query)
        else:
            res = qa(query)
            answer, docs = res['result'], res['source_documents']

        # Print the relevant sources used for the answer
            # print("hahah\n")
            print(res['result'])
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
            # print(res)
    
if __name__ == "__main__":
    main()
