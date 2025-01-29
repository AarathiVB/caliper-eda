import streamlit as st

st.set_page_config(
    page_title="Home - CALIPER",
    page_icon="chart_with_upwards_trend",
    layout="centered",
    )

st.title('CALIPER: A toolbox for exploratory analysis of experiment data')

st.markdown("""
\n
📁 Data Upload Section\n
This is your hub for uploading datasets, validating schemas, and previewing your data before the analysis begins. Click on the "Data" section to get started.

📊 Data Analysis Section\n
Choose from our suite of explorative data analysis functions to gain valuable insights from your datasets. Visit the "Analysis" section after you uploaded your data.
\n
Happy data exploring,

Aarathi Vijayachandran, Friedrich Schwager
""")