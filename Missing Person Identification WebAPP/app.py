import streamlit as st
import pickle
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time,copy,random
import os
from fpdf import FPDF
import base64

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

if 'report_pdf' not in st.session_state:
    st.session_state['report_pdf'] = FPDF()

def identify(inp_img):
    start = time.time()
    try:
        gray=cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
        inp_faces = st.session_state['face_cascade'].detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        ct = 0
        new_img = copy.deepcopy(inp_img)
        person_dir = {}
        for (x, y, w, h) in inp_faces:
            ct+=1
            input_face_region = inp_img[y:y+h, x:x+w]
            img = (input_face_region / 255.).astype(np.float32)
            img = cv2.resize(img, dsize = (224,224))
            embedding_vector = st.session_state['embedder'].predict(np.expand_dims(img, axis=0),verbose=0)[0]
            embv_scaled = st.session_state['utils']['scaler'].transform([embedding_vector])
            embv_pca = st.session_state['utils']['pca'].transform(embv_scaled)
            probabs = st.session_state['face_rec_model'].predict(embv_pca,verbose=False)
            top_n_probabs = probabs[0][np.argsort(probabs).reshape(-1)[::-1][:]]
            top_n_names = st.session_state['utils']['le'].inverse_transform(np.argsort(probabs).reshape(-1)[::-1][:])
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

def login():
    # Create an empty container
    placeholder = st.empty()

    # Insert a form in the container
    with placeholder.form("login"):
        st.markdown("#### Enter your credentials 🪪")
        email = st.text_input("Email :e-mail:")
        password = st.text_input("Password 	:key:", type="password")
        submit = st.form_submit_button("Login")

    actual_email = "email"         #needs to be modified later on during db integration
    actual_password = "password"

    if submit and email == actual_email and password == actual_password:
        # If the form is submitted and the email and password are correct,
        # clear the form/container and display a success message
        st.session_state['login_submit_status'] = True
        placeholder.empty()
        st.success("Login successful :unlock:")
    elif submit and (email != actual_email or password != actual_password):
        st.error("Login failed")
    else:
        pass

def refresh_user_page():
    st.session_state['login_submit_status'] = False
    st.session_state['identify_btn_status'] = False
    st.session_state['person_dir'] = {}
    st.session_state['dis_img'] = ""
    st.session_state['dis_cp'] = ""
    st.session_state['user_report'] = {}
    st.session_state['report_pdf'] = FPDF()

def create_report_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download Report</a>'

img_dir = r"D:\Users\DELL\Desktop\Major Project\105_classes_pins_dataset"

choice = st.sidebar.selectbox("Welcome!",["About","User Login"])

if choice == "About":
    st.title("Missing Person Identification System :sleuth_or_spy:")

    st.subheader("Final Year (Sem 8) - Major Project :computer: :bookmark_tabs:")

    st.subheader("Abstract:")
    st.success("""Our work offers a novel method for identifying
missing persons that makes use of recent developments in
face recognition technology. Conventional techniques depend on
converting images into embeddings and then figuring out how far
apart picture embeddings are from one another by calculating
distances between them, making it computationally demanding,
particularly while dealing with huge databases. Our process
involves first extracting faces from images using OpenCV’s face
extractor module, and then using VGGFace to turn those faces
into embeddings. However, we employ Principal Component
Analysis (PCA) to further reduce their dimensionality to 128-
dimensional vectors rather than directly comparing embeddings.
On the basis of these reduced embeddings, a Dense Neural
Network is then trained. With this method, the traditional O(n)
time for classification is greatly reduced to O(1). Our tests show
that our methodology outperforms numerous other methods and
reaches an excellent accuracy of 98.75%. This simplified method
gives encouraging outcomes in situations involving missing persons
identification in addition to increasing efficiency.
Index Terms—Face Extraction, Face Recognition, CNN, Deep
Learning, PCA, Missing person Identification""")

    st.subheader("Authors :busts_in_silhouette::")

    col1,col2,col3 = st.columns(3)

    with col1:
        with st.popover("Dhruv Limbani"):
            st.info("""Student,
                Department of Data Science
                and Business Systems,
                School of Computing,
                SRM Institute of Science and
                Technology, Kattankulathur
                603203, Chennai, India""")
    with col2:
        with st.popover("Abhishek Barhate"):
            st.info("""Student,
                Department of Data Science
                and Business Systems,
                School of Computing,
                SRM Institute of Science and
                Technology, Kattankulathur
                603203, Chennai, India""")
    with col3:
        with st.popover("Hemavathi D"):
            st.info("""Associate Professor,
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
        st.title("Upload Image :outbox_tray:")
        uploaded_file = st.file_uploader("Only jpg or jpeg allowed")
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image[...,::-1], caption="Uploaded Image")
            st.info("Please Enter the details of person found :pencil:")
            with st.form("details"):
                name_of_person = st.text_input("Name:")
                age = st.number_input("Age:",step=1)
                gender = st.radio("Gender:",('Male', 'Female', 'Other'))
                address = st.text_input("Address:")
                loaction = st.text_input("Location of identification:")
                submit_btn = st.form_submit_button("Submit")
                if submit_btn:
                    st.success("Details Saved!")
            identify_btn = st.button("Identify :mag:")
            if identify_btn:
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
                                        st.image(ver_img_path, caption=name)
                                        if st.checkbox(f"Select {name}",key=name+str(id)):
                                            if id not in st.session_state['user_report']['ids'].keys():
                                                st.session_state['user_report']['ids'][id] = {name}
                                            else:
                                                st.session_state['user_report']['ids'][id].add(name)
                    with col2:
                        r = st.button("Refresh :arrows_counterclockwise:")
        
            chk_rep = st.button("Check My Report :clipboard:")
            if chk_rep:
                st.info("Captured Image:")
                st.image(st.session_state['user_report']['captured_img'])
                shape = st.session_state['user_report']['captured_img'][...,::-1].shape
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Times', 'B', 20)
                pdf.cell(0, 10, "User Report", align='C')
                pdf.ln()
                pdf.set_font('Times', 'B', 16)
                pdf.cell(40, 10, "Captured Image:")
                cv2.imwrite("report_img.png", st.session_state['user_report']['captured_img'][...,::-1])
                pdf.ln()
                if(shape[0]>shape[1]):
                    pdf.image("report_img.png",h=150)
                else:
                    pdf.image("report_img.png",w=150)
                pdf.ln()
                pdf.cell(40, 10, "Personal details of Reported Person: ")
                pdf.ln()
                pdf.set_font('Times', 'B', 12)
                pdf.cell(15, 10, "Name: ")
                pdf.set_font('Times', '', 12)
                pdf.cell(40, 10, name_of_person)
                pdf.ln()
                pdf.set_font('Times', 'B', 12)
                pdf.cell(10, 10, "Age: ")
                pdf.set_font('Times', '', 12)
                pdf.cell(40, 10, str(age))
                pdf.ln()
                pdf.set_font('Times', 'B', 12)
                pdf.cell(18, 10, "Gender: ")
                pdf.set_font('Times', '', 12)
                pdf.cell(40, 10, gender)
                pdf.ln()
                pdf.set_font('Times', 'B', 12)
                pdf.cell(18, 10, "Address: ")
                pdf.set_font('Times', '', 12)
                pdf.cell(80, 10, address)
                pdf.ln()
                pdf.set_font('Times', 'B', 12)
                pdf.cell(50, 10, "Location of Identification: ")
                pdf.set_font('Times', '', 12)
                pdf.cell(80, 10, loaction)
                pdf.add_page()
                for id,names in st.session_state['user_report']['ids'].items():
                    st.info(f"Potential matches reported by user for ID-{id}:")
                    pdf.ln()
                    pdf.set_font('Times', 'B', 16)
                    pdf.cell(40, 10, f"Potential matches reported by user for ID-{id}:")
                    col4, col5 = st.columns([0.5,5])
                    with col5:
                        for i,nm in enumerate(names):
                            st.write(nm)
                            pdf.ln()
                            pdf.set_font('Times',"", 12)
                            pdf.cell(40, 10, f"\t\t\t\t\t{i+1}) {nm}")
                    st.session_state['report_pdf'] = pdf
            sbt_rep = st.button("Submit Report :incoming_envelope:")
            if sbt_rep:
                html = create_report_download_link(st.session_state['report_pdf'].output(dest="S").encode("latin-1"), "report")
                st.markdown(html, unsafe_allow_html=True)
        
        logout = st.button("Logout :end:")

        if st.session_state['login_submit_status'] and logout:
            st.session_state['login_submit_status'] = False  # requires double click to logout 
            refresh_user_page()
