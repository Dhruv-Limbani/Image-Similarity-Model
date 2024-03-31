import streamlit as st
import pickle
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time,copy,random
import os

if 'embedder' not in st.session_state:
    st.session_state['embedder'] = load_model("vgg_face_embedder.h5")

if 'face_rec_model' not in st.session_state:
    st.session_state['face_rec_model'] = load_model('face_rec_96.h5')

if 'face_cascade' not in st.session_state:
    st.session_state['face_cascade'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if 'utils' not in st.session_state:
    with open("scaler_pca_ledl","rb") as f:
        scaler,pca,le = pickle.load(f)
    st.session_state['utils'] = {
        'scaler' : scaler,
        'pca' : pca,
        'le' : le
    }

if 'login_submit_status' not in st.session_state:
    st.session_state['login_submit_status'] = False

if 'identify_btn_status' not in st.session_state:
    st.session_state['identify_btn_status'] = False

if 'person_dir' not in st.session_state:
    st.session_state['person_dir'] = {}

if 'dis_img' not in st.session_state:
    st.session_state['dis_img'] = ""

if 'dis_cp' not in st.session_state:
    st.session_state['dis_cp'] = ""

if 'user_report' not in st.session_state:
    st.session_state['user_report'] = {}

def identify(inp_img):
    start = time.time()
    try:
        gray=cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
        inp_faces = st.session_state['face_cascade'].detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        ct = 0
        new_img = copy.deepcopy(inp_img)
        # print("Maximum Probabilities:")
        person_dir = {}
        for (x, y, w, h) in inp_faces:
            ct+=1
            input_face_region = inp_img[y:y+h, x:x+w]
            img = (input_face_region / 255.).astype(np.float32)
            img = cv2.resize(img, dsize = (224,224))
            embedding_vector = st.session_state['embedder'].predict(np.expand_dims(img, axis=0),verbose=0)[0]
            embv_scaled = st.session_state['utils']['scaler'].transform([embedding_vector])
            embv_pca = st.session_state['utils']['pca'].transform(embv_scaled)
            #name = le.inverse_transform(mlp.predict(embv_pca))[0][5:]
            probabs = st.session_state['face_rec_model'].predict(embv_pca,verbose=False)
            top_n_probabs = probabs[0][np.argsort(probabs).reshape(-1)[::-1][:]]
            top_n_names = st.session_state['utils']['le'].inverse_transform(np.argsort(probabs).reshape(-1)[::-1][:])
            # max_indices = np.argsort(probabs)[-5:]
            # names = st.session_state['utils']['le'].inverse_transform(max_indices)
            # print(names)
            # if np.max(probabs) > 0.90:
            #     name = st.session_state['utils']['le'].inverse_transform([np.argmax(probabs)])[0][5:]
            # else:
            #     name = 'not recognized'
            # print(np.max(probabs),"-", name)
            cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,255,0),2)
            new_img = cv2.putText(new_img,str(ct),(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2,cv2.LINE_4)
            st.session_state['person_dir'][ct] = {}
            for i in range(len(top_n_names)):
                st.session_state['person_dir'][ct][top_n_names[i][5:].title()] = round(top_n_probabs[i]*100,4)
        new_img = new_img[...,::-1]
        end = time.time()
        cp = f"Total Execution Time: {'{:.2f}'.format((end-start)*1000)} ms"
        return new_img, cp
    except:
        st.error("Error")

def identify_one(inp_img):
    start = time.time()
    try:
        gray=cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
        inp_faces = st.session_state['face_cascade'].detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        new_img = copy.deepcopy(inp_img)
        for (x, y, w, h) in inp_faces:
            input_face_region = inp_img[y:y+h, x:x+w]
            img = (input_face_region / 255.).astype(np.float32)
            img = cv2.resize(img, dsize = (224,224))
            embedding_vector = st.session_state['embedder'].predict(np.expand_dims(img, axis=0),verbose=0)[0]
            embv_scaled = st.session_state['utils']['scaler'].transform([embedding_vector])
            embv_pca = st.session_state['utils']['pca'].transform(embv_scaled)
            probabs = st.session_state['face_rec_model'].predict(embv_pca,verbose=False)
            if np.max(probabs) > 0.90:
                name = st.session_state['utils']['le'].inverse_transform([np.argmax(probabs)])[0][5:]
            else:
                name = 'not recognized'
            cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,255,0),2)
            new_img = cv2.putText(new_img,name,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2,cv2.LINE_4)
        new_img = new_img[...,::-1]
        st.image(new_img,caption=cp)
        end = time.time()
        cp = f"Total Execution Time: {'{:.2f}'.format((end-start)*1000)} ms"
    except:
        st.error("Error")

def login():
    # Create an empty container
    placeholder = st.empty()

    # Insert a form in the container
    with placeholder.form("login"):
        st.markdown("#### Enter your credentials")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    actual_email = "email"         #needs to be modified later on during db integration
    actual_password = "password"

    if submit and email == actual_email and password == actual_password:
        # If the form is submitted and the email and password are correct,
        # clear the form/container and display a success message
        st.session_state['login_submit_status'] = True
        placeholder.empty()
        st.success("Login successful")
    elif submit and (email != actual_email or password != actual_password):
        st.error("Login failed")
    else:
        pass

def refresh_user_page():
    st.session_state['identify_btn_status'] = False
    st.session_state['person_dir'] = {}
    st.session_state['dis_img'] = ""
    st.session_state['dis_cp'] = ""

img_dir = r"D:\Users\DELL\Desktop\Major Project\105_classes_pins_dataset"

choice = st.sidebar.selectbox("Welcome!",["About","User Login"])

if choice == "About":
    st.title("Missing Person Identification System")

    st.subheader("Final Year (Sem 8) - Major Project")

    st.write("""Our work offers a novel method for identifying missing persons that makes use of recent developments in face recognition technology. Conventional techniques depend on figuring out how far apart picture embeddings are from one another by calculating distances between the embeddings, which can be computationally demanding, particularly when dealing with big datasets. Our process involves first extracting faces from images using OpenCV’s face extractor module, and then using VGGFace to turn those faces into embeddings. However, we employ Principal Component Analysis (PCA) to further reduce their dimensionality to 128-dimensional vectors rather than directly comparing embeddings. On the basis of these reduced embeddings, a Dense Neural Network is then trained. With this method, the traditional O(n) time for classification is greatly reduced to O(1). Our tests show that our methodology outperforms numerous other methods and reaches an excellent accuracy of 96%. This simplified method gives encouraging outcomes in situations involving missing persons identification in addition to increasing efficiency""")

    st.subheader("Author Details:")

    col1,col2,col3 = st.columns(3)

    fs = """------------Dhruv Limbani-------------
                    ----------------Student------------------
                    ---Department of Data Science----
                    -------and Business Systems,-------
                    --------School of Computing,--------
                    ---SRM Institute of Science and----
                    ---Technology, Kattankulathur-----
                    ------603203, Chennai, India--------"""

    with col1:
        st.write("""Dhruv Limbani, Student,
    Department of Data Science
    and Business Systems,
    School of Computing,
    SRM Institute of Science and
    Technology, Kattankulathur
    603203, Chennai, India""")
    with col2:
        st.write("""Abhishek Barhate, Student,
    Department of Data Science
    and Business Systems,
    School of Computing,
    SRM Institute of Science and
    Technology, Kattankulathur
    603203, Chennai, India""")
    with col3:
        st.write("""Hemavathi D, Associate Professor,
    Department of Data Science
    and Business Systems,
    School of Computing,
    SRM Institute of Science and
    Technology, Kattankulathur
    603203, Chennai, India""")

    st.subheader("""Department of Data Science and Business Systems, School of Computing, SRM Institute of Science and Technology, Kattankulathur 603203, Chennai, India""")

if choice == "User Login":
    
    if not st.session_state['login_submit_status']:
        login()

    if st.session_state['login_submit_status']:
        st.title("Upload Image")
        uploaded_file = st.file_uploader("Only jpg or jpeg allowed")
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image[...,::-1], caption="Uploaded Image")
            with st.form("details"):
                name = st.text_input("Name:")
                age = st.number_input("Age:",step=1)
                gender = st.radio("Gender:",('Male', 'Female', 'Other'))
                address = st.text_input("Address:")
                loaction = st.text_input("Location of identification:")
                submit_btn = st.form_submit_button("Submit")
            
            
            identify_btn = st.button("Identify")
            if identify_btn:
                refresh_user_page()
                st.session_state['identify_btn_status'] = True
                st.session_state['dis_img'], st.session_state['dis_cp'] = identify(image)
                st.session_state['user_report'] = {"captured_img" : " ", "ids" : {}}

            if st.session_state['identify_btn_status']:
                st.session_state['user_report']['captured_img'] = st.session_state['dis_img']           
                st.image(st.session_state['dis_img'], caption=st.session_state['dis_cp'])
                iden_opts = st.select_slider("Number of potential matches to be retrieved", [1,5,10,20,50,100,'all'])
                
                if iden_opts:
                    col1,col2 = st.columns([7,1])
                    with col1:
                        with st.container(height=1000,border=True):
                            for id in st.session_state['person_dir'].keys():
                                st.info(f"Id - {id}")
                                if iden_opts != 'all':
                                    lst = list(st.session_state['person_dir'][id].items())[:iden_opts]
                                else:
                                    lst = list(st.session_state['person_dir'][id].items())
                                for name, score in lst:
                                    with st.popover(f"{score}% ---> {name}"):
                                        path = os.path.join(img_dir,f"pins_{name}")
                                        ver_img = random.choice(os.listdir(path))
                                        ver_img_path = os.path.join(path,ver_img)
                                        st.image(ver_img_path, caption=st.session_state['dis_cp'])
                                        if st.checkbox(f"Select {name}",key=name+str(id)):
                                            if id not in st.session_state['user_report']['ids'].keys():
                                                st.session_state['user_report']['ids'][id] = {name}
                                            else:
                                                st.session_state['user_report']['ids'][id].add(name)
                    with col2:
                        r = st.button("Refresh")
        
        chk_rep = st.button("Check My Report")
        if chk_rep:
            st.info("Captured Image:")
            st.image(st.session_state['user_report']['captured_img'])
            for id,names in st.session_state['user_report']['ids'].items():
                st.info(f"Potential matches selected for ID-{id}:")
                col4, col5 = st.columns([0.5,5])
                with col5:
                    for name in names:
                        st.write(name)
        sbt_rep = st.button("Submit Report")
        if sbt_rep:
            st.success("Success! Your identification report has been sent to the nearest authority.")
        logout = st.button("Logout")

        if st.session_state['login_submit_status'] and logout:
            st.session_state['login_submit_status'] = False  # requires double click to logout 
            refresh_user_page()

    

    
    
        

    
