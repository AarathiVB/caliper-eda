import streamlit as st
import io
import dython.nominal as associations
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import ast
import numpy as np
from data_access_functions import (
    get_merged_distribution_data,
    get_merged_data,
    get_columns_for_task,
    PARTICIPANT_ID,
    SITE,
)

# To ignore all warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Analysis - CALIPER",
    page_icon="chart_with_upwards_trend",
    layout="centered",
)

# site string constants
MAGDEBURG = "Magdeburg"
ILMENAU = "Ilmenau"
CHEMNITZ = "Chemnitz"

with open("style.css", "r") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

st.title("Data Analysis Hub")
st.write(
    "Here you can dive deep into the insights hidden within your uploaded data. We've organized the analysis tools into tabs, each dedicated to a specific function. To get started, simply select the analysis function you're interested in."
)
st.header("How it Works")
st.markdown(
    """
1. **Choose a Tab:** Navigate through the tabs to explore different analysis functions. Each tab is designed to address specific aspects of your data.
2. **Specify Input:** For every analysis function, you'll find a set of input parameters that tailor the analysis to your needs.
3. **Uncover Insights:** Once you've selected the analysis function and specified the inputs, the results will be displayed in visualizations that are downloadble.
"""
)

(
    get_distribution_summary,
    get_pairwise_correlation,
) = st.tabs(
    [
        "Univariate Analysis: Get distribution summary",
        "Bivariate Analysis: Get pairwise correlation",
    ]
)

def create_correlation_matrix(dataframe, text_color):
    object_column_names = []
    number_column_names = []

    # Calculate the correlation matrix
    # get all nominal columns
    # Select columns of type "object"
    object_columns = dataframe.select_dtypes(include=["object"])
    object_column_names = object_columns.columns.tolist()

    number_columns = dataframe.select_dtypes(include=["int64", "float64"])
    number_column_names = number_columns.columns.tolist()

    # QUICKFIX: increase the figure size if the number of columns is large --> replace with filter logic for attributes
    if len(number_column_names) > 20:
        fig, ax = plt.subplots(figsize=(45, 30))
    else:
        fig, ax = plt.subplots(figsize=(15, 10))

    r = associations.associations(
        dataframe,
        ax=ax,
        nominal_columns=object_column_names,
        numerical_columns=number_column_names,
        nom_nom_assoc="cramer",
        num_num_assoc="pearson"
    )

    # Get the current figure
    fig = plt.gcf()

    # Set the title - What is seen (corrl matrix for categorical attributes over all participants)
    title = f"Correlation matrix for task(s): {task} and participant(s): {participant}"
    fig.suptitle(title, fontsize=14, fontweight="bold", color=text_color, x=0.23, ha = "left" )

    # Style of captions:
    # title: what kind of diagram, which task, which participant
    # description with shown measures and selected sites

    # adjust description according to shown attributes --> numerical = Pearson's R etc.
    description = ""
    file_name_addition = ""
    if len(number_column_names) > 0:
        if len(object_column_names) > 0:
            file_name_addition = "combined"
            description = f"""
                Displayed measures: Pearson's R for num-num, Correlation Ratio for cat-num, Cramer's V for cat-cat \n 
                Chosen sites: {', '.join(selected_sites)}
            """
        else:
            file_name_addition = "numerical"
            description = f"""
                Displayed measures: Pearson's R for num-num \n 
                Chosen sites: {', '.join(selected_sites)}
            """
    elif len(object_column_names) > 0:
        file_name_addition = "categorical"
        description = f"""
            Displayed measures: Cramers's V for cat-cat \n 
            Chosen sites: {', '.join(selected_sites)}
        """

    # Add a description - description positioning
    count_object_column_names = len(object_column_names)
    count_number_column_names = len(number_column_names)
    description_x_coordinate = 0
    
    if count_object_column_names < 5 and count_number_column_names <5:
        description_x_coordinate = 0.19
    else:
        description_x_coordinate = 0.175
    if len(number_column_names) > 20:
        description_x_coordinate = 0.21
    
    fig.text(
        x=description_x_coordinate,
        y=0.89,
        s=description,
        ha="left",
        linespacing=0.5,
        fontsize=12,
        color=text_color,
        multialignment="left",
    )
    
    # save figure
    plot_bytes = io.BytesIO()
    plt.savefig(plot_bytes, format="png")

    # change appearance for dark themed UI
    fig.patch.set_alpha(0.0)
    axes = fig.get_axes()
    for axis in axes:
        axis.tick_params(axis="x", colors=text_color)
        axis.tick_params(axis="y", colors=text_color)

    st.pyplot(fig)

    plot_bytes = io.BytesIO()
    plt.savefig(plot_bytes, format="png")
    # Add a download button for the plot
    file_name = f"CALIPER_{file_name_addition}_correlation_heatmap_task_{task}_participant_all.png"
    label = f"Download {file_name_addition} plot"
    st.download_button(
        key=file_name_addition,
        label=label,
        data=plot_bytes,
        file_name=file_name,
        mime="image/png",
    )
    
def correlation(dataframe, task, participant, selected_sites, text_color):
    # get dataframe with all columns for task X
    task_dataframe = get_columns_for_task(dataframe, task)

    st.header("Results")
    
    # if one participant for all tasks
    if task == "all" and participant != "all":
        # get single participant line
        participant_split = list(participant)

        # Look up the row
        row = task_dataframe.loc[
            (task_dataframe[SITE] == participant_split[0])
            & (task_dataframe[PARTICIPANT_ID] == int(participant_split[1]))
        ]

        # Filter the DataFrame to get the columns that do not start with a "T" followed by two numbers
        sociodemographic_columns = row.filter(regex="^(?!T[0-9]{2}).*")
        st.write(f"Sociodemographic data for participant: {participant}")
        st.dataframe(sociodemographic_columns, hide_index=True)

        # Filter the DataFrame to get the columns that start with a "T" followed by two numbers
        task_columns = row.filter(regex="^T[0-9]{2}")

        rows = []
        new_column_names = []
        i = 0
        # get columns --> all that start with T01, all that start with T02 ... list of lists!
        while True:
            i += 1
            single_task_columns = []
            if i < 10:
                single_task_columns = task_columns.filter(regex="^T0" + str(i))
                if i == 1:
                    new_column_names.append("Task")
                    for col in single_task_columns:
                        # strip beginning task number
                        col = col.replace("T01 ", "")
                        new_column_names.append(col)
            else:
                single_task_columns = task_columns.filter(regex="^T" + str(i))
            if len(single_task_columns.columns) == 0:
                break
            first_row = single_task_columns.iloc[0].tolist()
            # add index number
            first_row.insert(0, i)
            rows.append(first_row)

        # build new dataframe T = index of outer list, columns, values are inner lists
        task_dataframe = pd.DataFrame(rows, columns=new_column_names)
        st.write(f"All task results for participant: {participant}")
        st.dataframe(task_dataframe, hide_index=True)
    else:
        # remove "Site" column
        task_dataframe.drop(SITE, axis=1, inplace=True)
        task_dataframe.drop(PARTICIPANT_ID, axis=1, inplace=True)
        st.write(f"The subset for task(s): {task} and participant(s): {participant}")
        st.dataframe(task_dataframe, hide_index=True)

    # TODO: if time is left: Give a unfoldable list of all attributes that can be chosen for the correlation matrix (FOR ALL, ALL case)
    # TODO: if time is left: scatterplot for num-num
    
    st.subheader("Correlation matrices")
    
    with st.expander("Displayed measures"):
        # give credit to dython package
        st.markdown("""
            The displayed correlation matrices show all possible attribute combinations and were created with the Python package Dython. It was written by Shaked Zychlinski in 2018 and uses the following measures to calculate the correlation/association between attributes: [[Zychlinski2018][Dython2018]](#sources)
            """
        )
        st.markdown(
            """
            **Pearson's R** is a measure of the linear correlation between two numerical attributes, with an R value ranging from -1 to 1. A positive value indicates a positive linear correlation, a negative value signifies a negative correlation, and an R value close to 0 suggests a weak or no linear correlation between the attributes [[Baker2019][UniNewcastle]](#sources).
            """
        )
        # Value ranges and meanings
        data = {
        "Pearson's R": ["Perfect negative linear correlation", "Strong negative linear correlation", "Moderate negative linear correlation","Weak negative linear correlation", "No correlation","Weak positive linear correlation", "Moderate positive linear correlation","Strong positive linear correlation", "Perfect positive linear correlation"],
        "Range": ["-1", "-0.8 > r > -1", "-0.4 > r >= -0.8", "0 > r >= -0.4", "0", "0.4 > r > 0", "0.8 > r >= 0.4", "1 > r >= 0.8", "1"]
        }
        st.image("./images/strength_of_correlation_university_of_newcastle.png", caption="Strength of correlation [UniNewcastle]")
        
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(data)

        # Display the DataFrame
        st.dataframe(df, hide_index=True)
        st.caption("Value ranges for Pearons's R [UniNewcastle]")
        
        st.markdown("""
            The **Correlation Ratio** assesses the association between a categorical and a numerical attribute. It provides a value between 0 and 1, where 0 implies no association, and 1 signifies a perfect association. Higher values indicate a stronger relationship between the categorical and numerical variables [[Hazewinkel2013]](#sources). It helps to answer the following question "Given a continuous number, how well can you know to which category it belongs to?" [[Zychlinski2018]](#sources)
          """
        )
        
        st.markdown("""
            **Cramer's V** is a measure of association between two categorical variables. It produces values between 0 and 1, where 0 indicates no association, and 1 represents a perfect association. The interpretation is that higher values suggest a stronger association between the categorical variables in the analysis [[Sun2010]](#sources).
            """)
        # Value ranges and meanings
        data = {
        "Cramer's V": ["Small Effect", "Medium Effect", "Large Effect"],
        "Range": [".07 > v >= .21", ".21 > v >= .35", "v > .35"]
        }

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(data)

        # Display the DataFrame
        st.dataframe(df, hide_index=True)
        st.caption("Value ranges for Cramer's V [Sun2010]")
        
    with st.expander("Interpreation of visualized matrices"):
        st.write("There are three different matrices. One for numerical attribtes only and the correlation between them, one for categorical attributes only and the association between them and one for all attributes that contain correlation and association values.")
        st.write("If the categorical matrix is missing, it means, that for the retrieved data, there was only one categorical attribute and no pairwise association could be determined.")
        st.markdown("""
                    If the columns and rows for an attribute are grey and contain the label **SV**, it means, that the underlying attribute had no variance, and no correlation/association could be determined.
        """)
        st.write(
            "The correlation/association values that are particularly intriguing are those associated to the task's correctness and all other attributes. These values offer insights into the primary factors (predictors) on which correctness predominantly depends."
        )
        st.markdown("""
            You can use the displayed data above to verify the results. If e.g. a negative Pearson's R value is present for two numerical attributes, than there is an inverse relationship: when one variable goes up, the other tends to go down, and vice versa [[UniNewcastle]](#sources).
        """
        )
        st.markdown("""
            "The balance in the value distribution of variables can influence correlation measures, particularly in the context of skewed or unbalanced distributions. Correlation measures, such as Pearson's correlation coefficient, are sensitive to outliers and extreme values [[Kim2015]](#sources)."
        """
        )
        st.markdown("""
            "In statistics you can never prove that there is a relationship between a pair of variables but the strength of the relationship gives an indication of the likelihood of a dependence [[Baker2019]](#sources)."
        """
        )
        st.markdown("""
            "When the Pearson's correlation coefficient R (or R2) is \"R = 0.964. You can say that 93% (R2 = 0.9642 = 0.93) of the variation in outcome is explained by the predictor variable\" [[Baker2019]](#sources)."
        """
       )
        
    st.markdown("<h4>Numerical attributes only</h4>", unsafe_allow_html=True)
    # transform ordinal column --> certainty --> convert to numerical by mapping to values
    mapping = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
    selected_col = task_dataframe.filter(like="Certainty")
    for col in selected_col:
        task_dataframe[col] = task_dataframe[col].map(mapping)
    # Select numerical columns
    numerical_cols = task_dataframe.select_dtypes(include=["int64", "float64"]).columns
    # Create the new DataFrame
    task_dataframe_num = task_dataframe[numerical_cols]
    create_correlation_matrix(dataframe=task_dataframe_num,text_color=text_color)

    # Select categorical columns
    categorical_cols = task_dataframe.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 1:
        # Create the new DataFrame
        task_dataframe_cat = task_dataframe[categorical_cols]
        st.markdown("<h4>Categorical attributes only</h4>", unsafe_allow_html=True)
        create_correlation_matrix(dataframe=task_dataframe_cat, text_color=text_color)

    # TODO if time is left --> correlation ratio only for all num to correctness --> single line
    # TODO if time is left --> correlation ratio only for all cat to certainty --> single line

    st.markdown("<h4>Numerical-Categorical combined</h4>", unsafe_allow_html=True)
    create_correlation_matrix(task_dataframe, text_color)

def fd_preprocessing(x_axis_attr_org, x_axis_attr, df_merged):
    if 'Good/Scrap' in x_axis_attr_org:
        task_id = x_axis_attr_org[-2:]
        df_merged = df_merged[df_merged['Task ID'] == 'T' + task_id]
        x_axis_attr = 'Good/Scrap'
    
    if 'Certainty' in x_axis_attr_org:
        task_id = x_axis_attr_org[-2:]
        df_merged = df_merged[df_merged['Task ID'] == 'T' + task_id]
        x_axis_attr = 'Certainty'

    # Group by the specified column and count unique participant IDs in each group    
    result_df = df_merged.groupby(x_axis_attr)['Participant ID'].nunique().reset_index()
    result_df.columns = [x_axis_attr, 'Participant Count']
    result_df = result_df.sort_values(by=x_axis_attr)

    # Convert x-axis values to capitalized case
    result_df[x_axis_attr] = result_df[x_axis_attr].str.title()

    return result_df, x_axis_attr

def create_fd_all_sites(df_merged_m, df_merged_c, df_merged_i):
    options1 = [f'Good/Scrap Response for Cylinder Task {i:02}' for i in range(1, 17)]
    options2 = [f'Certainty of Response for Cylinder Task {i:02}' for i in range(1, 17)]
    options3 = ['Gender', 'Mother Tongue', 'Left/Right Handed', 'Program of Study']
    options = options1 + options2 + options3
    options.insert(0, 'Not Selected')
    x_axis_attr = st.selectbox('Select a categorical attribute for X-axis', options)
    if x_axis_attr.lower() != 'not selected':
        x_axis_attr_org = x_axis_attr

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(16, 10))

        # Set the width of the bars
        bar_width = 0.2

        data_present_m = False
        data_present_c = False
        data_present_i = False

        if not isinstance(df_merged_m, str):
            result_df_m, x_axis_attr2 = fd_preprocessing(x_axis_attr_org, x_axis_attr, df_merged_m)
            positions_m = np.arange(len(result_df_m[x_axis_attr2]))
            ax.bar(positions_m, result_df_m['Participant Count'], width=bar_width, color='#17a4f1')
            data_present_m = True

        if not isinstance(df_merged_c, str):
            result_df_c, x_axis_attr2 = fd_preprocessing(x_axis_attr_org, x_axis_attr, df_merged_c)
            positions_c = np.arange(len(result_df_c[x_axis_attr2])) + bar_width
            ax.bar(positions_c, result_df_c['Participant Count'], width=bar_width, color='#ffbf65')
            data_present_c = True
                
        if not isinstance(df_merged_i, str):
            result_df_i, x_axis_attr2 = fd_preprocessing(x_axis_attr_org, x_axis_attr, df_merged_i)
            positions_i = np.arange(len(result_df_i[x_axis_attr2])) + 2 * bar_width
            ax.bar(positions_i, result_df_i['Participant Count'], width=bar_width, color='#b860d9')
            data_present_i = True

        # Plotting the bar charts in a single plot
        st.markdown(f"<h1 style='text-align: center; color: white; font-size: 24px;'>Frequency Distribution for {x_axis_attr_org} (All Available Sites)</h1>", unsafe_allow_html=True)

        # Add y-axis values on top of the bars
        for p in ax.patches:
            yval = round(p.get_height(), 2)
            ax.text(p.get_x() + p.get_width() / 2, yval + 0.005, f"{yval}", ha='center', va='bottom', fontsize=12)

        if data_present_c == True:
            x_position = positions_c
            result_df_plot = result_df_c
        elif data_present_m == True:
            x_position = positions_m
            result_df_plot = result_df_m
        elif data_present_i == True:
            x_position = positions_i
            result_df_plot = result_df_i
        
        # Customize the plot
        ax.set_xticks(x_position)
        ax.set_xticklabels(result_df_plot[x_axis_attr2], rotation=45, ha='right', fontsize=18)
        ax.set_xlabel(x_axis_attr_org, fontsize=22)
        ax.set_ylabel('Participant Count', fontsize=22)
        ax.tick_params(axis='y', labelsize=18)
        plt.title(f'Frequency Distribution for {x_axis_attr_org} (All Available Sites)', fontsize=25)

        # Add space above the highest bar
        max_height = ax.get_ylim()[1]
        ax.set_ylim(0, max_height + 5)

        # Add legend inside plot on top right
        ax.legend(loc='upper right', labels=['Magdeburg', 'Chemnitz', 'Ilmenau'], fontsize='large')
        
        plt.tight_layout()

        # Save the Matplotlib figure to a BytesIO object
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')

        st.pyplot(fig)        

        # Add a download button for the plot
        file_name = f"Frequency_Distribution_{x_axis_attr_org}_All_Sites.png"
        st.download_button(
            label="Download Plot",
            data=plot_bytes.getvalue(),
            file_name=file_name,
            mime="image/png"
        )

def create_fd(df_merged, x_axis_attr, selected_site):
    if selected_site == 'magdeburg':
        plot_color = '#17a4f1'
    elif selected_site == 'chemnitz':
        plot_color = '#ffbf65'
    elif selected_site == 'ilmenau':
        plot_color = '#b860d9'

    x_axis_attr_org = x_axis_attr

    result_df, x_axis_attr2 = fd_preprocessing(x_axis_attr_org, x_axis_attr, df_merged)
   
    # Find mode and its corresponding x-axis value
    mode_value = df_merged[x_axis_attr2].mode()

    # Plotting the bar chart
    st.markdown(f"<h1 style='text-align: center; color: white; font-size: 24px;'>Frequency Distribution for {x_axis_attr_org} ({selected_site.capitalize()})</h1>", unsafe_allow_html=True)
    # Bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(16, 10))
    bars = ax.bar(result_df[x_axis_attr2], result_df['Participant Count'], color=plot_color)
    plt.xticks(rotation=45, ha='right')
    ax.set_xlabel(x_axis_attr_org, fontsize=22)
    ax.set_ylabel('Participant Count', fontsize=22)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    plt.title(f'Frequency Distribution for {x_axis_attr_org} ({selected_site.capitalize()})', fontsize=25)
    # Display y-axis values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontsize=12)

    # Add some space at the top of bars
    max_yval = result_df['Participant Count'].max()
    ax.set_ylim(0, max_yval + 3)  # Adjust the value as needed
    # Add label for Mode in the top right corner
    mode_label = f'Mode = {max_yval} ({mode_value[0].title()})'
    ax.text(0.97, 0.98, mode_label, ha='right', va='top', transform=ax.transAxes, fontsize=24, color='red')

    plt.tight_layout()

    # Display the figure
    st.pyplot(fig)

    # Save the Matplotlib figure to a BytesIO object
    plot_bytes = io.BytesIO()
    plt.savefig(plot_bytes, format='png')

    # Add a download button for the plot
    file_name = f"Frequency_Distribution_{x_axis_attr_org}_Plot.png"
    st.download_button(
        label="Download Plot",
        data=plot_bytes.getvalue(),
        file_name=file_name,
        mime="image/png"
    )

def create_error(row):
    if row['Good/Scrap'] == 'good' and row['Ground Truth'] == 'good':
        return '0'
    elif row['Good/Scrap'] == 'good' and row['Ground Truth'] == 'scrap':
        return '1'
    elif row['Good/Scrap'] == 'scrap' and row['Ground Truth'] == 'scrap':
        return '0'
    elif row['Good/Scrap'] == 'scrap' and row['Ground Truth'] == 'good':
        return '1'
    else:
        return 'unknown'  
    
def error_preprocessing(df_error_rate, sort_choice):
    df_error_rate['Error'] = df_error_rate.apply(create_error, axis=1)

    # Pivot the DataFrame to get participant IDs as columns and task IDs as rows
    df_error_rate2 = df_error_rate.pivot(index='Task ID', columns='Participant ID', values='Error')

    # Add a column 'Error Rate for each Task'
    unique_participant_count = df_error_rate['Participant ID'].nunique()        
    df_error_rate2['Error Rate'] = df_error_rate2.eq('1').sum(axis=1)
    df_error_rate2['Error Rate'] = df_error_rate2['Error Rate'] / unique_participant_count

    if 'error' in sort_choice.lower():
        df_error_rate2 = df_error_rate2.sort_values(by='Error Rate', ascending=False)
    else:
        df_error_rate2 = df_error_rate2.sort_values(by='Task ID', ascending=True)
    
    return df_error_rate2, unique_participant_count

def error_rate_description(unique_participant_count, selected_site):
    st.markdown(f"<h1 style='text-align: center; color: white; font-size: 25px;'>Error Rate Table ({selected_site.capitalize()})</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; color: white; font-size: 21px;'>How to interpret the Error Rate Table?</h1>", unsafe_allow_html=True)
    st.markdown(f"""
1. **Columns represent participants with IDs from P01 to P{unique_participant_count}** 
2. **Rows represent the cylinder tasks with IDs from T01 to T16**
3. **If the participant's response to a cylinder task matches with the ground truth value of the task, the cell is given the value '0', which means error is absent** 
4. **If the participant's response to a cylinder task does not match with the ground truth value of the task, the cell is given the value '1', which means error is present** 
5. **Error Rate = Count of error present / Total number of participants**
6. **HIGHER the Error Rate, TOUGHER the cylinder task**
""")

def calc_error_rate_all_sites(df_error_rate_m, df_error_rate_c, df_error_rate_i):

    options = ['Error Rate', 'Task ID']
    options.insert(0, 'Not Selected')
    sort_choice = st.selectbox('Select Sorting Criteria', options)

    if sort_choice.lower() != 'not selected':

        # Create a single figure and axes
        fig, ax = plt.subplots(figsize=(30, 15))

        # Plot the first bar chart
        if not isinstance(df_error_rate_m, str):
            df_error_rate2_m, unique_participant_count_m = error_preprocessing(df_error_rate_m, sort_choice)
            df_error_rate2_m.plot(kind='bar', y='Error Rate', ax=ax, position=0, width=0.3, color='#17a4f1', legend=False)

        # Plot the second bar chart
        if not isinstance(df_error_rate_c, str):
            df_error_rate2_c, unique_participant_count_c = error_preprocessing(df_error_rate_c, sort_choice)
            df_error_rate2_c.plot(kind='bar', y='Error Rate', ax=ax, position=1, width=0.3, color='#ffbf65', legend=False)

        # Plot the third bar chart
        if not isinstance(df_error_rate_i, str):
            df_error_rate2_i, unique_participant_count_i = error_preprocessing(df_error_rate_i, sort_choice)
            df_error_rate2_i.plot(kind='bar', y='Error Rate', ax=ax, position=2, width=0.3, color='#b860d9', legend=False)

        # Customize the plot
        ax.set_xlabel('Task ID', fontsize=22)
        ax.set_ylabel('Error Rate', fontsize=22)
        ax.set_title('Error Rate vs Task ID (All Available Sites)', fontsize=25)
        plt.xticks(rotation=45, ha='right', fontsize=18)
        plt.yticks(fontsize=18)  
       
        # Add y-axis values on top of the bars
        for p in ax.patches:
            yval = round(p.get_height(), 2)
            ax.text(p.get_x() + p.get_width() / 2, yval + 0.005, f"{yval}", ha='center', va='bottom', fontsize=12)

        # Add some space at the end of the x-axis
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 0.6)

        # Add space above the highest bar
        max_height = ax.get_ylim()[1]
        ax.set_ylim(0, max_height + 0.1)

        # Add legend inside plot on top right
        ax.legend(loc='upper right', labels=['Magdeburg', 'Chemnitz', 'Ilmenau'], fontsize='large')
        # Display the figure
        st.pyplot(fig)

        # Save the Matplotlib figure to a BytesIO object
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')

        # Add a download button for the plot
        file_name = f"Error_Rate_Plot_All_Sites.png"
        st.download_button(
            label="Download Plot",
            data=plot_bytes.getvalue(),
            file_name=file_name,
            mime="image/png"
        )

def calc_error_rate(df_error_rate, selected_site):
    if selected_site == 'magdeburg':
        plot_color = '#17a4f1'
    elif selected_site == 'chemnitz':
        plot_color = '#ffbf65'
    elif selected_site == 'ilmenau':
        plot_color = '#b860d9'

    options = ['Error Rate', 'Task ID']
    options.insert(0, 'Not Selected')
    sort_choice = st.selectbox('Select Sorting Criteria', options)

    if sort_choice.lower() != 'not selected':

        df_error_rate2, unique_participant_count = error_preprocessing(df_error_rate, sort_choice)

        error_rate_description(unique_participant_count, selected_site)

        # Display the table using streamlit
        st.table(df_error_rate2)

        # Add a download button
        excel_file = io.BytesIO()
        df_error_rate2.to_excel(excel_file, index=True)
        excel_file.seek(0)
        st.download_button(
            label="Download Table",
            data=excel_file,
            file_name=f"Error_Rate_Table_{selected_site.capitalize()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown(f"<h1 style='text-align: center; color: white; font-size: 25px;'>'Error Rate vs Task ID' Bar Chart ({selected_site.capitalize()})</h1>", unsafe_allow_html=True)
        # Create and display a bar chart
        fig, ax = plt.subplots(figsize=(16, 10))
        df_error_rate2.plot(kind='bar', y='Error Rate', ax=ax, color=plot_color, legend=False)
        ax.set_xlabel('Task ID', fontsize=22)
        ax.set_ylabel('Error Rate', fontsize=22)
        ax.set_title(f'Error Rate vs Task ID ({selected_site.capitalize()})', fontsize=25)
        plt.xticks(rotation=45, ha='right', fontsize=18)
        plt.yticks(fontsize=18)
        # Add y-axis values on top of the bars
        for p in ax.patches:
            yval = round(p.get_height(), 2)
            ax.text(p.get_x() + p.get_width() / 2, yval + 0.005, f"{yval}", ha='center', va='bottom', fontsize=12)

        st.pyplot(fig)

        # Save the Matplotlib figure to a BytesIO object
        plot_bytes = io.BytesIO()
        plt.savefig(plot_bytes, format='png')

        # Add a download button for the plot
        file_name = f"Error_Rate_Plot_{selected_site.capitalize()}.png"
        st.download_button(
            label="Download Plot",
            data=plot_bytes.getvalue(),
            file_name=file_name,
            mime="image/png"
        )

        # Identify and print the task ID(s) with the highest error rate
        max_error_rate = df_error_rate2['Error Rate'].max()
        tasks_with_max_error_rate = df_error_rate2[df_error_rate2['Error Rate'] == max_error_rate].index.tolist()
        st.markdown(f"<p style='font-size: 22px; color: white;'>TOUGHEST TASK(S) = Task(s) with the highest error rate =  {', '.join(tasks_with_max_error_rate)}</p>", unsafe_allow_html=True)

        # Identify and print the task ID(s) with the lowest error rate
        min_error_rate = df_error_rate2['Error Rate'].min()
        tasks_with_min_error_rate = df_error_rate2[df_error_rate2['Error Rate'] == min_error_rate].index.tolist()
        st.markdown(f"<p style='font-size: 22px; color: white;'>EASIEST TASK(S) = Task(s) with the lowest error rate =  {', '.join(tasks_with_min_error_rate)}</p>", unsafe_allow_html=True)

def calc_box_plot(df_box_plot, selected_attr, selected_participant, selected_task, selected_site, selected_p_t):
    if selected_site == 'magdeburg':
        plot_color = '#17a4f1'
    elif selected_site == 'chemnitz':
        plot_color = '#ffbf65'
    elif selected_site == 'ilmenau':
        plot_color = '#b860d9'

    # Convert relevant columns to numeric if needed
    numeric_cols = ['Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']
    df_box_plot[numeric_cols] = df_box_plot[numeric_cols].apply(pd.to_numeric, errors='coerce')

    if selected_p_t.lower() == 'all participants':
        numeric_cols = ['Age', 'Total Time for all 16 Tasks per Participant']
    if selected_p_t.lower() == 'all tasks':
        numeric_cols = ['Diameter', 'Time for each Task', 'EDA Grand Mean']
    if ((selected_p_t.lower() == 'all participants') and (selected_task.lower() == 'all tasks')) or \
        ((selected_p_t.lower() == 'all tasks') and (selected_participant.lower() == 'all participants')):
        numeric_cols = ['Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']
    
    if (selected_participant.lower() != 'all participants') and (selected_participant.lower() != 'not selected'):
        df_box_plot = df_box_plot[df_box_plot['Participant ID'] == selected_participant]
   
    if (selected_task.lower() != 'all tasks') and (selected_task.lower() != 'not selected'):
        df_box_plot = df_box_plot[df_box_plot['Task ID'] == selected_task]

    if (selected_attr.lower() == 'all attributes'):
        df_box_plot = df_box_plot[numeric_cols]
    else:
        numeric_cols = [selected_attr]
        df_box_plot = df_box_plot[numeric_cols]

    # Calculate summary statistics
    summary_stats = df_box_plot.describe(percentiles=[.25, .5, .75])

    # Calculate interquartile range
    summary_stats.loc['IQR'] = summary_stats.loc['75%'] - summary_stats.loc['25%']

    # Calculate range of values
    summary_stats.loc['range'] = summary_stats.loc['max'] - summary_stats.loc['min']
    
    # Detect outliers
    outliers = {}
    for col in numeric_cols:
        Q1 = summary_stats.loc['25%', col]
        Q3 = summary_stats.loc['75%', col]
        IQR = summary_stats.loc['IQR', col]
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = list(set(df_box_plot[(df_box_plot[col] < lower_bound) | (df_box_plot[col] > upper_bound)][col]))

    summary_stats = summary_stats.applymap(lambda x: f"{int(x):,}" if x.is_integer() else f"{x:.4f}")
    # Rename the rows
    summary_stats = summary_stats.rename(index={
        'count': 'Count',
        'mean': 'Mean',
        'std': 'Standard Deviation',
        'min': 'Minimum Value',
        '25%': '25th Percentile',
        '50%': '50th Percentile or Median',
        '75%': '75th Percentile',
        'max': 'Maximum Value',
        'IQR': 'Inter-Quartile Range',
        'range': 'Range of Values'
    })
    # Display summary statistics
    st.subheader("Summary Statistics:")
    if selected_p_t.lower() == 'all participants':
        detail_str = f"Participant: All Participants, Task: {selected_task.title()}, Attribute: {selected_attr}"
        st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>{detail_str}</h1>", unsafe_allow_html=True)
    if selected_p_t.lower() == 'all tasks':
        detail_str = f"Task: All Tasks, Participant: {selected_participant.title()}, Attribute: {selected_attr}"
        st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>{detail_str}</h1>", unsafe_allow_html=True)
    st.table(summary_stats)

    # Download summary statistics as Excel
    excel_file = io.BytesIO()
    summary_stats.to_excel(excel_file, index=True)
    excel_file.seek(0)
    st.download_button(
        label="Download Summary Statistics",
        data=excel_file,
        file_name="summary_statistics.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Display separate box plots for each column
    st.subheader("Box Plots:")
    if selected_p_t.lower() == 'all participants':
        detail_str = f"Participant: All Participants, Task: {selected_task.title()}"
        st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>{detail_str}</h1>", unsafe_allow_html=True)
    if selected_p_t.lower() == 'all tasks':
        detail_str = f"Task: All Tasks, Participant: {selected_participant.title()}"
        st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>{detail_str}</h1>", unsafe_allow_html=True)
    
    for col in numeric_cols:
        plt.figure(figsize=(16, 10))
        box_plot = sns.boxplot(data=df_box_plot[col], orient='v', color=plot_color, width=0.3, linewidth=1.5, fliersize=3, boxprops=dict(edgecolor='black'))
        plt.title(f"Box Plot [{detail_str}, Attribute: {col}] ({selected_site.capitalize()}) \n Displayed Measures: 25th Percentile (Q1), 50th Percentile or Median (Q2), 75th Percentile (Q3), Inter-Quartile Range (IQR), Outliers", fontsize=25)
        # Adjust font size on the y-axis tick labels
        plt.yticks(fontsize=18)
        plt.ylabel(col, fontsize=22)

        plt.tight_layout()

        # Save the box plot as PNG
        box_plot_file = io.BytesIO()
        plt.savefig(box_plot_file, format='png', bbox_inches='tight')
        box_plot_file.seek(0)

        st.pyplot(plt)

        # Download box plot as PNG
        st.download_button(
            label=f"Download Box Plot for {col}",
            data=box_plot_file,
            file_name=f"Box_Plot_{col}.png",
            mime="image/png"
        )

    # Display outliers
    st.subheader("Outliers:")
    # Remove key-value pairs where the value is an empty list
    outliers_dict = {key: value for key, value in outliers.items() if value}
    
    if outliers_dict != {}:
        st.write(outliers_dict)

        outliers_dict = {key: str(value) for key, value in outliers_dict.items()}
        
        for key, value in outliers_dict.items():
            outliers_dict[key] = ast.literal_eval(value)
        
        outliers_df = pd.DataFrame({key: pd.Series(value) for key, value in outliers_dict.items()})

        # Download outliers as Excel
        excel_outliers_file = io.BytesIO()
        outliers_df.to_excel(excel_outliers_file, index=False)
        excel_outliers_file.seek(0)

        st.download_button(
            label="Download Outliers",
            data=excel_outliers_file,
            file_name="Outliers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
    else:
        st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>No outliers for {selected_attr}</h1>", unsafe_allow_html=True)

def boxplot_preprocessing(df_box_plot, selected_p_t, selected_participant, selected_task, selected_attr):

    numeric_cols = ['Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']
    df_box_plot[numeric_cols] = df_box_plot[numeric_cols].apply(pd.to_numeric, errors='coerce')

    if selected_p_t.lower() == 'all participants':
        numeric_cols = ['Age', 'Total Time for all 16 Tasks per Participant']
    if selected_p_t.lower() == 'all tasks':
        numeric_cols = ['Diameter', 'Time for each Task', 'EDA Grand Mean']
    if ((selected_p_t.lower() == 'all participants') and (selected_task.lower() == 'all tasks')) or \
        ((selected_p_t.lower() == 'all tasks') and (selected_participant.lower() == 'all participants')):
        numeric_cols = ['Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']
    
    if ((selected_participant.lower() != 'all participants') and (selected_participant.lower() != 'not selected')):
        df_box_plot = df_box_plot[df_box_plot['Participant ID'] == selected_participant]
    
    if ((selected_task.lower() != 'all tasks') and (selected_task.lower() != 'not selected')):
        df_box_plot = df_box_plot[df_box_plot['Task ID'] == selected_task]
    
    if (selected_attr.lower() == 'all attributes'):
        df_box_plot = df_box_plot[numeric_cols]
    else:
        numeric_cols = [selected_attr]
        df_box_plot = df_box_plot[numeric_cols]

    return df_box_plot

def create_boxplot_all_sites(df_merged_m, df_merged_c, df_merged_i):
    selected_participant = 'Not Selected'
    selected_task = 'Not Selected'
    st.markdown("""
                <h1 style='text-align: left; color: white; font-size: 18px;'>
                    The 'Mean, Median, Percentiles, Interquartile Range (IQR), Range, Standard Deviation, Outliers' are calculated based on the following criteria:
                </h1>
                <ul style='text-align: left; color: white; font-size: 18px;'>
                    <li>All Participants, One Task, One or All Attributes</li>
                    <li>All Tasks, One Participant, One or All Attributes</li>
                    <li>All Participants, All Tasks, One or All Attributes</li>
                </ul>
                """, unsafe_allow_html=True)

    options4 = ['All Participants', 'All Tasks']
    options4.insert(0, 'Not Selected')
    selected_p_t = st.selectbox("Select 'All Participants' or 'All Tasks'", options4)

    unique_participant_count_m = 0
    unique_participant_count_c = 0
    unique_participant_count_i = 0
    if not isinstance(df_merged_m, str):
        unique_participant_count_m = df_merged_m['Participant ID'].nunique()
    if not isinstance(df_merged_c, str):
        unique_participant_count_c = df_merged_c['Participant ID'].nunique()
    if not isinstance(df_merged_i, str):
        unique_participant_count_i = df_merged_i['Participant ID'].nunique()
    
    max_participant_count = max(unique_participant_count_m, unique_participant_count_c, unique_participant_count_i)

    if selected_p_t.lower() == 'all participants':
        options3 = [f'T{i:02}' for i in range(1, 17)]
        options3.insert(0, 'All Tasks')
        selected_task = st.selectbox('Select one or all cylinder tasks', options3)

        if ((selected_p_t.lower() == 'all participants') and (selected_task.lower() == 'all tasks')):
            options1 = ['Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']
            options1.insert(0, 'All Attributes')
            selected_attr = st.selectbox('Select one or all numerical attributes', options1)
        else:
            options1 = ['Age', 'Total Time for all 16 Tasks per Participant']
            options1.insert(0, 'All Attributes')
            selected_attr = st.selectbox('Select one or all numerical attributes', options1)

    if (selected_p_t.lower() == 'all tasks'):
        options2 = [f'P{i:02}' for i in range(1, max_participant_count + 1)]
        options2.insert(0, 'All Participants')
        selected_participant = st.selectbox('Select one or all participants', options2)

        if ((selected_p_t.lower() == 'all tasks') and (selected_participant.lower() == 'all participants')):
            options1 = ['Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']
            options1.insert(0, 'All Attributes')
            selected_attr = st.selectbox('Select one or all numerical attributes', options1)
        else:
            options1 = ['Diameter', 'Time for each Task', 'EDA Grand Mean']
            options1.insert(0, 'All Attributes')
            selected_attr = st.selectbox('Select one or all numerical attributes', options1)

    if selected_p_t.lower() != 'not selected':
        
        available_sites = []
        if not isinstance(df_merged_m, str):
            df_box_plot_m = df_merged_m[['Participant ID', 'Task ID', 'Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']]
            df_box_plot_m = boxplot_preprocessing(df_box_plot_m, selected_p_t, selected_participant, selected_task, selected_attr)
            available_sites.append(('Site M', df_box_plot_m))
        
        if not isinstance(df_merged_c, str):
            df_box_plot_c = df_merged_c[['Participant ID', 'Task ID', 'Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']]
            df_box_plot_c = boxplot_preprocessing(df_box_plot_c, selected_p_t, selected_participant, selected_task, selected_attr)
            available_sites.append(('Site C', df_box_plot_c))

        if not isinstance(df_merged_i, str):
            df_box_plot_i = df_merged_i[['Participant ID', 'Task ID', 'Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']]
            df_box_plot_i = boxplot_preprocessing(df_box_plot_i, selected_p_t, selected_participant, selected_task, selected_attr)
            available_sites.append(('Site I', df_box_plot_i))

        if selected_p_t.lower() == 'all participants':
            detail_str = f"Participant: All Participants, Task: {selected_task.title()}"
            st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>{detail_str}</h1>", unsafe_allow_html=True)
        if selected_p_t.lower() == 'all tasks':
            detail_str = f"Task: All Tasks, Participant: {selected_participant.title()}"
            st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>{detail_str}</h1>", unsafe_allow_html=True)
        
        if selected_attr.lower() != 'all attributes':
            df_combined = pd.concat([df_box_plot[selected_attr].rename(site) for site, df_box_plot in available_sites], axis=1)

            # Define custom colors for each site
            colors = {'Site M': '#17a4f1', 'Site C': '#ffbf65', 'Site I': '#b860d9'}

            # Create a grouped box plot
            plt.figure(figsize=(16, 10))
            sns.boxplot(data=df_combined, width=0.5, palette=colors)
            # Adjust font sizes
            plt.title(f"Box Plot [{detail_str}, Attribute: {selected_attr}] (All Available Sites) \n Displayed Measures: 25th Percentile (Q1), 50th Percentile or Median (Q2), 75th Percentile (Q3), Inter-Quartile Range (IQR), Outliers", fontsize=25)
            
            plt.xlabel("Site", fontsize=22)
            plt.ylabel(selected_attr, fontsize=22)

            # Adjust x-axis and y-axis tick label font sizes
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.tight_layout()

            # Display the figure
            st.pyplot(plt)

            # Save the box plot as PNG
            box_plot_file = io.BytesIO()
            plt.savefig(box_plot_file, format='png', bbox_inches='tight')  # Add bbox_inches='tight' parameter
            box_plot_file.seek(0)

            # Download box plot as PNG
            st.download_button(
                label=f"Download Box Plot for {selected_attr}",
                data=box_plot_file,
                file_name=f"Box_Plot_{selected_attr}_All_Sites.png",
                mime="image/png"
            )

        elif (selected_attr.lower() == 'all attributes'):

            if selected_p_t.lower() == 'all participants':
                all_attr = ['Age', 'Total Time for all 16 Tasks per Participant']
            if selected_p_t.lower() == 'all tasks':
                all_attr = ['Diameter', 'Time for each Task', 'EDA Grand Mean']
            if ((selected_p_t.lower() == 'all participants') and (selected_task.lower() == 'all tasks')) or \
                    ((selected_p_t.lower() == 'all tasks') and (selected_participant.lower() == 'all participants')):
                all_attr = ['Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']             
            
            for j, col in enumerate(all_attr):
                # Combine the data for all three sites
                df_combined = pd.concat([df_box_plot[col].rename(site) for site, df_box_plot in available_sites], axis=1)

                # Define custom colors for each site
                colors = {'Site M': '#17a4f1', 'Site C': '#ffbf65', 'Site I': '#b860d9'}

                # Create a grouped box plot
                plt.figure(figsize=(16, 10))
                sns.boxplot(data=df_combined, width=0.5, palette=colors)
                # Adjust font sizes
                plt.title(f"Box Plot [{detail_str}, Attribute: {col}] (All Available Sites) \n Displayed Measures: 25th Percentile (Q1), 50th Percentile or Median (Q2), 75th Percentile (Q3), Inter-Quartile Range (IQR), Outliers", fontsize=25)
                plt.xlabel("Site", fontsize=22)
                plt.ylabel(col, fontsize=22)

                # Adjust x-axis and y-axis tick label font sizes
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                plt.tight_layout()

                # Display the figure
                st.pyplot(plt)

                # Save the box plot as PNG
                box_plot_file = io.BytesIO()
                plt.savefig(box_plot_file, format='png', bbox_inches='tight')
                box_plot_file.seek(0)

                # Generate a unique key for the download button
                download_button_key = f"Download_Button_{col.replace(' ', '_')}_{j}"

                # Download box plot as PNG with unique key
                st.download_button(
                    label=f"Download Box Plot for {col}",
                    data=box_plot_file,
                    file_name=f"Box_Plot_{col.replace(' ', '_')}_All_Sites.png",
                    key=download_button_key,
                    mime="image/png"
                )


def calc_distribution_summary(stat_qty, selected_site, df_merged):

    if (stat_qty.lower() != 'not selected') and (selected_site.lower() in ['magdeburg', 'chemnitz', 'ilmenau']):
        if ('frequency distribution' in stat_qty.lower()):
            options1 = [f'Good/Scrap Response for Cylinder Task {i:02}' for i in range(1, 17)]
            options2 = [f'Certainty of Response for Cylinder Task {i:02}' for i in range(1, 17)]
            options3 = ['Gender', 'Mother Tongue', 'Left/Right Handed', 'Program of Study']
            options = options1 + options2 + options3
            options.insert(0, 'Not Selected')
            x_axis_attr = st.selectbox('Select a categorical attribute for X-axis', options)
            if x_axis_attr.lower() != 'not selected':
                create_fd(df_merged, x_axis_attr, selected_site)
        
        if ('error' in stat_qty.lower()):
            df_error_rate = df_merged[['Participant ID', 'Task ID', 'Good/Scrap', 'Ground Truth']]
            calc_error_rate(df_error_rate, selected_site)
        
        if ('median' in stat_qty.lower()):
            st.markdown("""
                <h1 style='text-align: left; color: white; font-size: 18px;'>
                    The 'Mean, Median, Percentiles, Interquartile Range (IQR), Range, Standard Deviation, Outliers' are calculated based on the following criteria:
                </h1>
                <ul style='text-align: left; color: white; font-size: 18px;'>
                    <li>All Participants, One Task, One or All Attributes</li>
                    <li>All Tasks, One Participant, One or All Attributes</li>
                    <li>All Participants, All Tasks, One or All Attributes</li>
                </ul>
                """, unsafe_allow_html=True)
            selected_participant = 'Not Selected'
            selected_task = 'Not Selected'

            options4 = ['All Participants', 'All Tasks']
            options4.insert(0, 'Not Selected')
            selected_p_t = st.selectbox("Select 'All Participants' or 'All Tasks'", options4)

            unique_participant_count = df_merged['Participant ID'].nunique()
                        
            if selected_p_t.lower() == 'all participants':
                options3 = [f'T{i:02}' for i in range(1, 17)]
                options3.insert(0, 'All Tasks')
                selected_task = st.selectbox('Select one or all cylinder tasks', options3)

                if ((selected_p_t.lower() == 'all participants') and (selected_task.lower() == 'all tasks')):
                    options1 = ['Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']
                    options1.insert(0, 'All Attributes')
                    selected_attr = st.selectbox('Select one or all numerical attributes', options1)
                else:
                    options1 = ['Age', 'Total Time for all 16 Tasks per Participant']
                    options1.insert(0, 'All Attributes')
                    selected_attr = st.selectbox('Select one or all numerical attributes', options1)

            if selected_p_t.lower() == 'all tasks':
                options2 = [f'P{i:02}' for i in range(1, unique_participant_count + 1)]
                options2.insert(0, 'All Participants')
                selected_participant = st.selectbox('Select one or all participants', options2)

                if ((selected_p_t.lower() == 'all tasks') and (selected_participant.lower() == 'all participants')):
                    options1 = ['Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']
                    options1.insert(0, 'All Attributes')
                    selected_attr = st.selectbox('Select one or all numerical attributes', options1)
                else:
                    options1 = ['Diameter', 'Time for each Task', 'EDA Grand Mean']
                    options1.insert(0, 'All Attributes')
                    selected_attr = st.selectbox('Select one or all numerical attributes', options1)
                
            if selected_p_t.lower() != 'not selected':
                df_box_plot = df_merged[['Participant ID', 'Task ID', 'Age', 'Diameter', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']]
                calc_box_plot(df_box_plot, selected_attr, selected_participant, selected_task, selected_site, selected_p_t)

with get_distribution_summary:
    st.header("Get Distribution Summary")
    # st.markdown("<h1 style='text-align: left; color: white; font-size: 21px;'>This section gives the following statistical quantities: Frequency Distribution with Mode, Mean, Median, Percentiles, Interquartile Range (IQR), Range, Standard Deviation, Outliers, Error Rate</h1>", unsafe_allow_html=True)
    st.markdown("""
                <h1 style='text-align: left; color: white; font-size: 25px;'>
                    This section gives the following statistical quantities:
                </h1>
                <ul style='text-align: left; color: white; font-size: 25px;'>
                    <li>Frequency Distribution with Mode</li>
                    <li>Mean</li>
                    <li>Median</li>
                    <li>Percentiles</li>
                    <li>Interquartile Range (IQR)</li>
                    <li>Range</li>
                    <li>Standard Deviation</li>
                    <li>Outliers</li>
                    <li>Error Rate</li>
                </ul>
                """, unsafe_allow_html=True)
    
    # Select Statistical Quantity
    options_stat_qty = ['Frequency Distribution with Mode', 'Mean, Median, Percentiles, Interquartile Range (IQR), Range, Standard Deviation, Outliers', 'Error Rate']
    options_stat_qty.insert(0, 'Not Selected')
    stat_qty = st.selectbox('Select Statistical Quantity', options_stat_qty)

    options_site = ['Magdeburg', 'Chemnitz', 'Ilmenau', 'All Sites']
    options_site.insert(0, 'Not Selected')
    selected_site = st.selectbox('Select Site', options_site)
    
    selected_site = selected_site.lower()

    if ('sites' not in selected_site.lower()) and ('selected' not in selected_site.lower()):
        df_merged = get_merged_distribution_data(selected_site)
        if (type(df_merged) is str) and ('not available' in df_merged.lower()):
            st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>{df_merged}</h1>", unsafe_allow_html=True)
        else:
            calc_distribution_summary(stat_qty, selected_site, df_merged)
    
    elif ('sites' in selected_site.lower()) and ('selected' not in selected_site.lower()):
        
        if ('error' in stat_qty.lower()):
            df_merged_m = get_merged_distribution_data('magdeburg')
            df_merged_c = get_merged_distribution_data('chemnitz')
            df_merged_i = get_merged_distribution_data('ilmenau')
            if (type(df_merged_m) is str):
                st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>Data for Magdeburg site is not available. Displaying the Error Rate for other sites.</h1>", unsafe_allow_html=True)
            if (type(df_merged_c) is str):
                st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>Data for Chemnitz site is not available. Displaying the Error Rate for other sites.</h1>", unsafe_allow_html=True)
            if (type(df_merged_i) is str):
                st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>Data for Ilmenau site is not available. Displaying the Error Rate for other sites.</h1>", unsafe_allow_html=True)
            try:
                df_error_rate_m = df_merged_m[['Participant ID', 'Task ID', 'Good/Scrap', 'Ground Truth']]
            except:
                df_error_rate_m = ""
            try:
                df_error_rate_c = df_merged_c[['Participant ID', 'Task ID', 'Good/Scrap', 'Ground Truth']]
            except:
                df_error_rate_c = ""
            try:
                df_error_rate_i = df_merged_i[['Participant ID', 'Task ID', 'Good/Scrap', 'Ground Truth']]
            except:
                df_error_rate_i = ""
        
            calc_error_rate_all_sites(df_error_rate_m, df_error_rate_c, df_error_rate_i)

        elif ('frequency' in stat_qty.lower()):
            df_merged_m = get_merged_distribution_data('magdeburg')
            df_merged_c = get_merged_distribution_data('chemnitz')
            df_merged_i = get_merged_distribution_data('ilmenau')
            if (type(df_merged_m) is str):
                st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>Data for Magdeburg site is not available. Displaying the Frequency Distribution for other sites.</h1>", unsafe_allow_html=True)
            if (type(df_merged_c) is str):
                st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>Data for Chemnitz site is not available. Displaying the Frequency Distribution for other sites.</h1>", unsafe_allow_html=True)
            if (type(df_merged_i) is str):
                st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>Data for Ilmenau site is not available. Displaying the Frequency Distribution for other sites.</h1>", unsafe_allow_html=True)
                            
            create_fd_all_sites(df_merged_m, df_merged_c, df_merged_i)

        elif ('median' in stat_qty.lower()):
            df_merged_m = get_merged_distribution_data('magdeburg')
            df_merged_c = get_merged_distribution_data('chemnitz')
            df_merged_i = get_merged_distribution_data('ilmenau')
            if (type(df_merged_m) is str):
                st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>Data for Magdeburg site is not available. Displaying the Box Plot for other sites.</h1>", unsafe_allow_html=True)
            if (type(df_merged_c) is str):
                st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>Data for Chemnitz site is not available. Displaying the Box Plot for other sites.</h1>", unsafe_allow_html=True)
            if (type(df_merged_i) is str):
                st.markdown(f"<h1 style='text-align: left; color: white; font-size: 21px;'>Data for Ilmenau site is not available. Displaying the Box Plot for other sites.</h1>", unsafe_allow_html=True)
                            
            create_boxplot_all_sites(df_merged_m, df_merged_c, df_merged_i)


with get_pairwise_correlation:
    st.header("Get pairwise correlation")
    
    with st.expander("What is a bivariate analysis, correlation and association?"):
        st.markdown("""
            In the following, we will look at the relationship (statistical dependence) between two variables with each other, i.e. how to perform a bivariate analysis. [[Cleff2015]](#sources)
            
            "When you are looking for a relationship between two numercial variables, such as age and EDA grand mean, then the test you use is called a **correlation**. If one or both of the variables are categorical, such as the task correctness (True or False), then the test is called an **association**.
            When you can phrase your hypothesis (question or hunch) in the following form, then you are talking about the relationship family of statistical analyses:" [[Baker2019]](#sources)

            - Is the EDA value of the participant related to the correctness of the task?
            - Are the age of a participant and the time to solve a task correlated?
            - Is the certainty of a participant associated with the task correctness?
        """
        )

    st.subheader("Parameter selection")
    
    # List of sites
    sites = [MAGDEBURG, ILMENAU, CHEMNITZ]

    # User selects one or multiple sites
    selected_sites = st.multiselect("Choose one or more experiment sites:", sites)

    # Check if any options were selected
    if len(selected_sites)!=0:
        get_site_m = MAGDEBURG in selected_sites
        get_site_c = CHEMNITZ in selected_sites
        get_site_i = ILMENAU in selected_sites

        # get data
        data = get_merged_data(get_site_m, get_site_c, get_site_i)
        # display retrieved data
        st.write("Retrieved data:")
        st.dataframe(data, hide_index=True)
        
        st.markdown("""
        Choose whether you want to find the relationship of the attributes:
        - over one task and all participants
        - over all tasks and one participant
        - or over all tasks and alls participants         
        """)

        task = "all"
        # get participants = ID + site
        participant_options = []
        if task == "all":
            participant_options = [
                f"{row['Site']}{row['Response ID']}" for _, row in data.iterrows()
            ]
        participant_options.insert(0, "all")
        # Create a selectbox with the title "Participant" and the options
        participant = st.selectbox("Participant", participant_options)

        # if one participant selected, only add "all" selection to tasks
        task_options = []
        if participant == "all":
            # Create a list of options from T01 to T16
            task_options = [f"T{i:02}" for i in range(1, 17)]
        # Add a placeholder to the list of options
        task_options.insert(0, "all")

        # Create a selectbox with the title "Task" and the options
        task = st.selectbox("Task", task_options)

        # Create a radio button for selecting the text color
        text_color = st.radio("Text color for the visualizations", ("white", "black"), index=0)

        # Create the button
        if st.button("Calculate"):
            correlation(data, task, participant, selected_sites, text_color)

    else:
        st.warning("No sites were selected. Please select at least one site to continue.")
        
    st.markdown("<h4>Sources</h4>", unsafe_allow_html=True)
    st.markdown("""
        [Cleff2015] Cleff, T. 2015. Deskriptive Statistik und Explorative Datenanalyse. page 73. Springer Gabler.  

        [Baker2019] Baker, L. 2019. Associations and Correlations. 1st edition. chapter 3,4. Packt Publishing.
        
        [UniNewscastle] Strength of Correlation. (n.d.). University of Newcastle. https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/strength-of-correlation.html.
        
        [Dython2018] Zychlinski, S. 2018. "Dython" GitHub Repository. https://github.com/shakedzy/dython.
                
        [Zychlinski2018] Zychlinski, S. 2018. "The Search for Categorical Correlation" Towards Data Science. https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9.
                
        [Hazewinkel2013] Hazewinkel, M. 2013. Encyclopaedia of Mathematics: Coproduct — Hausdorff—Young Inequalities, page 8. Springer US.
                
        [Sun2010] [Sun, S., Pan, W., & Wang, L. L. (2010, September 20). A Comprehensive Review of Effect Size Reporting and Interpreting Practices in Academic Journals in Education and Psychology. Journal of Educational Psychology. Advance online publication. page 7. doi: 10.1037/a0019507](https://www.researchgate.net/profile/Wei-Pan-8/publication/232485860_A_Comprehensive_Review_of_Effect_Size_Reporting_and_Interpreting_Practices_in_Academic_Journals_in_Education_and_Psychology/links/541858e30cf203f155ada928/A-Comprehensive-Review-of-Effect-Size-Reporting-and-Interpreting-Practices-in-Academic-Journals-in-Education-and-Psychology.pdf).
    
        [Kim2014] Kim, Y., Kim, T. H., & Ergün, T. (2015). The instability of the Pearson correlation coefficient in the presence of coincidental outliers. Finance Research Letters, 13, 243-257. 
    """)
