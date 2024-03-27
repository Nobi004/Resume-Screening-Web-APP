import streamlit as st
import nltk
import re
import pickle


#Loading models
knc = pickle.load(open('knc.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

nltk.download('stopwords')
from nltk.corpus import stopwords
def clean_resume(resume_text):
    #Compile patterns for URLs and emails to speed up cleaning process
    urls = re.compile(r'https?://\S+|www\.\S+')
    emails = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    #Remove URLs
    clean_text = urls.sub('',resume_text)

    #Remove emails
    clean_text = emails.sub('',clean_text)

    #Remove special characters (keep only words and with space)
    clean_text = re.sub(r'[^\w\s]','',clean_text)

    # Remove stop words by filtering the split words of the text
    stop_words = set(stopwords.words('english'))
    clean_text = ' '.join(word for word in clean_text.split() if word.lower() not in stop_words)

    return clean_text
#Web APP
def main():
    st.title("Resume Screening APP")
    upload_file = st.file_uploader('Upload your Resume',type=('txt','pdf'))
    if upload_file is not None :
        try :
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError :
            #IF UTF-8 decoding fails , try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

            cleaned_resume = clean_resume(resume_text)
            input_features = tfidf.transform([cleaned_resume])
            prediction_id = knc.predict(input_features)
            st.write("Predicted Category:",prediction_id)

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21:"SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }
            # Assuming prediction_id is a NumPy array
            prediction_id = tuple(prediction_id)
            print(prediction_id)
            category_name = category_mapping.get(prediction_id[0], "Unknown")

            st.write("Predicted Category:", category_name)


# python main
if __name__ == "__main__":
    main()
