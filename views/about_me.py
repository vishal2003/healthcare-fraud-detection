import streamlit as st

from forms.contact import contact_form


@st.dialog("Contact Me")
def show_contact_form():
    contact_form()


def app():
    col1, col2 = st.columns(2, gap='small', vertical_alignment='center')
    with col1:
        st.image('./assets/medium.jpg', width=230)
    with col2:
        st.title('Ayomitan Adesua')
        st.write(
            'Healthcare Data Analyst, health information manager and coding instructor for kids'
        )
        if st.button('✉️ Contact Me'):
            show_contact_form()

    # --- Experience and Qualification
    st.write('\n')
    st.subheader('Experience & Qualifications', anchor=False)
    st.write(
    '''
     - 5 Years experience extracting actional business insights from data
     - Strong hands-on experience and knowledge in Python, SQL, Tableau, SPSS, and Spreadsheets
     - Strong background in Biostatistical principles and their applications
     - Ability to take initiatives and strong communication skills
    '''
    )

    # --- SKILLS -- 
    st.write('\n')
    st.subheader('Hard Skills', anchor=False)
    st.write(
    '''
     - Programming: Python (Scikit-learn, Pandas, Seaborn), SQL, SPSS
     - Data Visualization: Tableau, Seaborn, and Excel
     - Modeling: Logistic Regression, Linear Regression, SGBoosting
     - Database: SQL Server, MySQL
    '''
    )

    st.write('\n')
    st.subheader('Soft Skills', anchor=False)
    st.write(
    '''
     - Problem-solving
     - Initiative
     - Communication
     - Leadership
     - Teamwork
    '''
    )

    


