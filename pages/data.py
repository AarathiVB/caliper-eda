import streamlit as st
import sqlite3
import pandas as pd
from streamlit_js_eval import streamlit_js_eval

# page settings - must come first
st.set_page_config(
    page_title="Data - CALIPER",
    page_icon="chart_with_upwards_trend",
    layout="centered",
    )
st.title('Data Upload')

ANSWERS = "Answers"
GROUND_TRUTH = "Ground Truth"
EDA = "EDA"

# TODO order the attribtues? --> First upper case exact matches, then Lower case matches
answers_columns = ['Response ID','Seed', 'Last page','Start language','time','good or scrap', 'date', 'Please enter your age', 'Please specify your program of study', 'is it good?', 'diameter', 'certain', 'Please specify your mother tongue', 'Are you left or right handed?','Do you have any further comments?','Please enter your gender']
answers_datatypes = ['int64','int64','int64','object','float64','object','object', 'int64', 'object', 'object','float64','object','object','object','object','object']
answers_schema = pd.DataFrame([answers_datatypes], columns=answers_columns)
answers_schema_display = answers_schema.replace('int64', 'Integer')
answers_schema_display.replace('float64', 'Float', inplace=True)
answers_schema_display.replace('object', 'String', inplace=True)

ground_truth_columns = ['Task ID', 'Ground Truth']
ground_truth_datatypes = ['int64', 'object']
ground_truth_schema = pd.DataFrame([ground_truth_datatypes], columns=ground_truth_columns)
ground_truth_schema_display = ground_truth_schema.replace('int64', 'Integer')
ground_truth_schema_display.replace('object', 'String', inplace=True)

eda_columns = ['Response ID', 'GrandMean']
eda_datatypes = ['int64', 'float64']
eda_schema = pd.DataFrame([eda_datatypes], columns=eda_columns)
eda_schema_display = eda_schema.replace('int64', 'Integer')
eda_schema_display.replace('float64', 'Float', inplace=True)

# Create an collapsible section for the data schema
st.header("Upload instructions")
with st.expander("Your data has to follow the given schema:"):
    # replace datatype Strings to make it easier to understand for the user
    st.write("Answers Schema")
    st.write("All attributes in the schema that are in lower case, are partially matched. E.g. There are various attributes containing the String \'time\' and all of them should have the data type Float.")
    st.dataframe(answers_schema_display, hide_index=True)
    st.write("Ground Truth Schema")
    st.dataframe(ground_truth_schema_display, hide_index=True)
    st.write("EDA Schema")
    st.dataframe(eda_schema_display, hide_index=True)

# Check the data types of the columns in your dataframe
def validate_data_types(table_type, df):
    if table_type == ANSWERS:
        schema = answers_schema
    elif table_type == GROUND_TRUTH:
        schema = ground_truth_schema
    elif table_type == EDA:
        schema = eda_schema
    
    # Iterate over the columns of the DataFrame
    for column_name in df.columns:
        for schemaColumnName in schema.columns:
            if schemaColumnName.lower() in column_name.lower():
                # Get the corresponding data type from the schema
                schema_data_type = schema[schemaColumnName].iloc[0]
                # Check if the datatype of the column matches the type given in the schema
                if df[column_name].dtype != schema_data_type:
                    st.error(f'Data rejected: The column "{column_name}" partially matches a schema column name but its datatype does not match the schema')
                break
            # TODO more validation? write special cases for format: DATE
        else:
            st.error(f'Data rejected: The column "{column_name}" does not match any schema column names')
            return False
    return True

with st.expander("Your file has to follow the given naming convention:"):
    st.write('The name of the CSV file containing the data is expected in one of the following three formats:  \n **answers-site-X.CSV**  \n  **ground-truth-site-X.CSV**  \n **eda-site-X.CSV**  \n Please replace X with the letter of your site (M, I or C).')

with st.expander("Once the file is uploaded, check all values of categorical attributes"):
    st.write('For categorical attributes (all String attributes) we can only verify the data type. But we can not validate the semantical correctness. Let us consider a small example:  \n For the question: \"Please specify your program of study.\" there could be multiple answers that are sementically equal. E.g. \"Computer Science\", \"computer science\", \"Informatik\" or also typos such as \"Computer Sience\". For the purpose of cleaning the data, we advise you to take these steps:  \n 1) Get all categorical values and check for semantic equalities  \n 2) If you find different values that semantically mean the same:  \n 2.a) Delete the old data  \n 2.b) Find a single value for the semantically equal values (e.g. Computer Science)  \n 2.c) Upload your updated data')

# data upload - database is created if not existing
conn = sqlite3.connect('database.db')
uploaded_files = st.file_uploader("Upload CSV data", type = "csv", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    filename = uploaded_file.name.lower()
    proceed = False
    if "answers-site-" in filename and filename.endswith(".csv") and len(filename) == 18: 
        tableType = ANSWERS
        proceed = True
    if "ground-truth-site-" in filename and filename.endswith(".csv") and len(filename) == 23: 
        tableType = GROUND_TRUTH
        proceed = True
    if "eda-site-" in filename and filename.endswith(".csv") and len(filename) == 14: 
        tableType = EDA
        proceed = True
    if proceed:
        df = pd.read_csv(uploaded_file)
        if validate_data_types(tableType, df):
            conn = sqlite3.connect('database.db')
            table_name = filename.split('.')[0] # Get the name of the uploaded file without the extension
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            st.success("Upload successful!", icon="✅")
    else:
        st.error("Please upload a file with the name format 'answers-site-X.csv', 'ground-truth-site-X.csv' or ' eda-site-X.csv' where X is a single letter representing your site (M, I or C).")

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
table_names = [table[0] for table in tables] # Extract the table name from the tuple
table_name = st.selectbox('Uploaded data', table_names)
# Store the current table name in the session state
if 'current_table' not in st.session_state:
   st.session_state['current_table'] = table_name
# Check if the table name has changed
if st.session_state['current_table'] != table_name:
    st.session_state['current_table'] = table_name
    st.session_state['viewing_table'] = False

if table_name:
    if st.button("View"):
        df = pd.read_sql_query(f"SELECT * from `{table_name}`", conn)
        st.dataframe(df,hide_index=True)
        st.session_state['viewing_table'] = True
        st.session_state['df'] = df # store df in session state

    if 'viewing_table' in st.session_state and st.session_state['viewing_table']:
        df = st.session_state['df'] # retrieve df from session state
        if len(df.select_dtypes(include=[object])) > 0:
            # give check function that retrieves categorical features
            if st.button("Get unique values for all categorical attributes"):
                # Create a dictionary of column names and unique values for columns of type "object"
                dict_unique_values = {column: df[column].unique().tolist() for column in df.select_dtypes(include=[object]).columns}
                st.write(dict_unique_values)
            
        if st.button("Delete"):
            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            conn.commit()
            table_name=None
            st.session_state['viewing_table'] = False
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
conn.close()