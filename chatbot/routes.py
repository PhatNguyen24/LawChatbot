from flask import render_template, redirect, url_for, request, flash, jsonify, Response
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI, LlamaCpp, GPT4All
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from datetime import datetime, timedelta
from chatbot.models import User, Topic, Conversation, Pdf, Feedback
import os
import openai
from dotenv import load_dotenv
from chatbot import app, db, bcrypt

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

persist_directory = os.environ.get('PERSIST_DIRECTORY')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
chromadb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

# model_type = os.environ.get('MODEL_TYPE')
model_n_ctx = os.environ.get('MODEL_N_CTX')
n_gpu_layers = os.environ.get('N_GPU_LAYERS')
n_batch = os.environ.get('N_BATCH')
callbacks = [StreamingStdOutCallbackHandler()]
retriever = chromadb.as_retriever(search_kwargs={'k': 1})   
template="""Use the following pieces of information to answer the user's question.
    If you dont know the answer just say you know, don't try to make up an answer. Please use Vietnamese to answer the question

    Context:{context}
    Question:{question}

    Only return the helpful answer below and nothing else
    Helpful answer
    """

qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])

def gen_prompt(docs, query) -> str:
    return f"""To answer the question please only use the Context given, nothing else.
Question: {query}
Context: {[doc.page_content for doc in docs]}
Answer:
"""

def prompt(query):
     docs = chromadb.similarity_search(query, k=4) # tìm kiếm 4 văn bản tương tự 
     prompt = gen_prompt(docs, query)
     return prompt

def stream(input_text):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": "You're an assistant."},
        {"role": "user", "content": f"{prompt(input_text)}"},
    ], stream=True, temperature=0.5)
    for line in completion:
        if 'content' in line['choices'][0]['delta']:
            yield line['choices'][0]['delta']['content']  #tương tự return, khác cái là trả về hết vòng lặp thì mới thoát khỏi hàm

@app.template_filter('add_hours')
def add_hours(dt):
    return dt + timedelta(hours=7)

def get_current_date():
    return {'current_date': datetime.today().strftime('%Y-%m-%d %H:%M:%S')}

app.context_processor(get_current_date) 

# Hàm chuyển đổi tên người dùng
def convert_name_to_abbreviation(name):
    words = name.split()
    abbreviation = ""

    if len(words) > 1:
        abbreviation = words[0][0].upper() + words[-1][0].upper()
    elif len(words) == 1:
        abbreviation = words[0][0].upper()

    return abbreviation

# ////////////////////

@app.route('/')
@app.route('/home')
def home():  
    return render_template('home_chatbot.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            next_page = request.args.get('next')
            if(not user.topics):
                topic = Topic(user_id = user.id)
                db.session.add(topic)
                db.session.commit()
            return redirect(url_for('admin', user=user.id)) if user.role == 'admin' else redirect(url_for('baseuser', user=user.id))
        else:
            flash('Đăng nhập không thành công, vui lòng kiểm tra lại tài khoản hoặc mật khẩu', 'danger')
    return render_template('login.html')

@app.route('/sigup', methods=["GET", "POST"])
def sigup():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Tài khoản đã tồn tại', 'warning')
        elif password != confirm_password:
            flash('Mật khẩu không hợp lệ', 'warning')
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            flash('Bạn đã tạo tài khoản thành công! Bạn có thể đăng nhập bây giờ', 'success')
            return redirect(url_for('login'))
    return render_template('sigup.html')

@app.route('/baseuser')
def baseuser():
    id_user = request.args.get('user')
    id_topic = request.args.get('topic')
    user = User.query.filter_by(id=id_user).first()
    new_name = convert_name_to_abbreviation(user.username)
    topic = Topic.query.filter_by(id=id_topic).first()
    return render_template('baseuser.html', user=user, topic = topic, new_name = new_name)

@app.route('/usernotaccount')
def usernotaccount():
    return render_template('usernotaccount.html')

@app.route('/admin')
def admin():
    users = User.query.filter_by(role='user').all()
    user_id = request.args.get('user')
    pdfs = Pdf.query.filter_by(user_id= user_id).all()
    topic = Topic.query.filter_by(user_id = user_id).first()

    #feedback
    user_emails = []
    user_chats = []
    bot_chats = []
    feedbacks = Feedback.query.all();
    for feedback in feedbacks:
        user_email = User.query.filter_by(id=feedback.user_id).first().email
        user_chat = Conversation.query.filter_by(id=feedback.conversation_id).first().user_chat
        bot_chat = Conversation.query.filter_by(id=feedback.conversation_id).first().bot_chat
        # user_emails.append(user_email)
        # user_chats.append(user_chat)
        # bot_chats.append(bot_chat)
        feedback.user_email = user_email
        feedback.user_chat = user_chat
        feedback.bot_chat = bot_chat
    return render_template('admin.html', users = users, topic = topic, pdfs = pdfs, user_id = user_id, user_emails = user_emails, user_chats = user_chats, bot_chats = bot_chats, feedbacks = feedbacks)

@app.route('/add_topic', methods=['POST'])
def add_topic():
    data = request.get_json()
    if 'user_id' in data:
        user_id = data['user_id']
        new_topic = Topic(user_id=user_id)
        db.session.add(new_topic)
        db.session.commit()
        return jsonify({'nofitication': 'Thêm topic thành công'})
    else:
        return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

@app.route('/rename_topic', methods=['POST'])
def rename_topic():
    
    data = request.get_json()
    if 'updatedTitle' in data:
        updatedTitle = data['updatedTitle']
        topic_id = data['topic_id']
        getTopic = Topic.query.filter_by(id = topic_id).first()
        getTopic.title = updatedTitle
        db.session.commit()
        return jsonify({'topic': topic_id})
    else:
        return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

@app.route('/delete_topic', methods=['POST'])
def delete_topic():
    data = request.get_json()
    if 'topic_id' in data:
        topic_id = data['topic_id']
        db.session.query(Conversation).filter_by(topic_id=topic_id).delete()
        db.session.query(Topic).filter_by(id=topic_id).delete()
        db.session.commit()
        last_topic_id = Topic.query.order_by(Topic.id.desc()).first().id
        return jsonify({'last_topic_id': last_topic_id})
    else:
        return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

@app.route('/add_conversation', methods=['GET', 'POST'])
def add_conversation():
    # data = request.get_json()
    if request.method == "POST":
        data = request.form
        if 'input_text' in data:
            input_text = data['input_text']
            model_type = data['model']
            print(model_type)
            # print(input_text)
            model_path = "chatbot/models/ggml-gpt4all-j-v1.3-groovy.bin"
            if 'topic_id' in data:
                topic_id = data['topic_id']
                model_type = data['model']
                if(model_type == "ChatGPT"):
                    # return Response(stream(input_text), mimetype='text/event-stream')
                    return Response(stream(input_text), mimetype='text/event-stream')
                elif (model_type == "Llama2"):
                    model_path = "chatbot/models/llama-2-7b-chat.Q4_K_M.gguf"
                    llm = LlamaCpp(
                        model_path=model_path,
                        n_gpu_layers=n_gpu_layers,
                        n_batch=n_batch,
                        callbacks=callbacks,
                        verbose=True,  # Verbose is required to pass to the callback manager
                        n_ctx=model_n_ctx
                    )
                else: 
                    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
                qa = RetrievalQA.from_chain_type(
                    llm=llm, 
                    chain_type="stuff", 
                    retriever=retriever, 
                    return_source_documents= True, 
                    chain_type_kwargs={'prompt': qa_prompt}
                )
                res = qa(input_text)
                conversation = Conversation( user_chat = input_text, bot_chat = res['result'], topic_id = topic_id)
                db.session.add(conversation)
                db.session.commit()
            # answer, docs = res['result'], res['source_documents']
                return Response(res['result'], mimetype='text/event-stream')
                # bot_chat = get_conversation_chain(vec)({"question": (prompt + user_chat)})
                # return jsonify({'user_chat': user_chat, 'bot_chat': bot_chat["answer"]})
            else: 
                if(model_type == "Llama2"):
                    model_path = "chatbot/models/llama-2-7b-chat.Q4_K_M.gguf"
                    llm = LlamaCpp(
                        model_path=model_path,
                        n_gpu_layers=n_gpu_layers,
                        n_batch=n_batch,
                        callbacks=callbacks,
                        verbose=True,  # Verbose is required to pass to the callback manager
                        n_ctx=model_n_ctx
                    )
                else:
                    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
                qa = RetrievalQA.from_chain_type(
                    llm=llm, 
                    chain_type="stuff", 
                    retriever=retriever, 
                    return_source_documents= True, 
                    chain_type_kwargs={'prompt': qa_prompt}
                )
                res = qa(input_text)
                # answer, docs = res['result'], res['source_documents']
                return Response(res['result'], mimetype='text/event-stream')
    # if 'topic_id' in data:
    #     


# @app.route('/completion', methods=['GET', 'POST'])
# def completion_api():
#     if request.method == "POST":
#         data = request.form
#         input_text = data['input_text']
#         return Response(stream(input_text), mimetype='text/event-stream')
#     else:
#         return Response(None, mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    user_id = request.form.get('user_id')
    if uploaded_file:
        file_name = uploaded_file.filename
        pdf_record = Pdf.query.filter_by(pdfname= file_name ).first()
        if pdf_record:
            flash('File đã tồn tại', 'danger')
            return 'File uploaded error!'
        else:
            new_pdf = Pdf(pdfname = file_name, user_id=user_id)
            db.session.add(new_pdf)
            db.session.commit()
            upload_path = os.path.join('.', 'data', uploaded_file.filename)
            uploaded_file.save(upload_path)
            return 'Tải file thành công!'
    else:
        return 'Không có file nào được chọn.'

@app.route('/delete_user', methods=['POST'])
def delete_user():
    data = request.get_json()
    if 'user_id' in data:
        user_id = data['user_id']
        # delete conversation
        topics_to_delete = db.session.query(Topic).filter_by(user_id=user_id).all()
        for topic in topics_to_delete:
            db.session.query(Conversation).filter_by(topic_id=topic.id).delete()
        # delete topic 
        db.session.query(Topic).filter_by(user_id=user_id).delete()
        # delete user 
        db.session.query(User).filter_by(id=user_id).delete()
        db.session.commit()
        return jsonify({'nofitication': 'Xoá người dùng thành công'})
    else:
        return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

@app.route('/delete_pdf', methods=['POST'])
def delete_pdf():
    data = request.get_json()
    if 'pdf_id' in data:
        pdf_id = data['pdf_id']
        # xoa file o thu muc
        pdf = db.session.query(Pdf).filter_by(id=pdf_id).first()
        file_path = os.path.join('.', 'chatPDF', 'pdf', pdf.pdfname)
        os.remove(file_path)
        # delete pdf
        db.session.query(Pdf).filter_by(id=pdf_id).delete()
        db.session.commit()
        return jsonify({'nofitication': 'Xoá file thành công'})
    else:
        return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

@app.route('/delete_feedback', methods=['POST'])
def delete_feedback():
    data = request.get_json()
    if 'feedback_id' in data:
        feedback_id = data['feedback_id']
        db.session.query(Feedback).filter_by(id=feedback_id).delete()
        db.session.commit()
        return jsonify({'nofitication': 'Xoá phản hồi thành công'})
    else:
        return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

@app.route('/add_feedback', methods=['POST'])
def add_feedback():
    data = request.get_json()
    if 'message' in data:
        message = data['message']
        user_id = data['user_id']
        conversation_id = data['conversation_id']
        feedback = Feedback( message = message, user_id = user_id, conversation_id = conversation_id)
        db.session.add(feedback)
        db.session.commit()
        return jsonify({'nofitication': 'Thêm phản hồi thành công'})
    else:
        return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

