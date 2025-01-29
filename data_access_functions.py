import sqlite3
import pandas as pd
import numpy as np
import re
import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")

# table name constants
ANSWERS_SITE_M = "answers-site-m"
EDA_SITE_M = "eda-site-m"
GROUND_TRUTH_SITE_M = "ground-truth-site-m"

ANSWERS_SITE_C = "answers-site-c"
EDA_SITE_C = "eda-site-c"
GROUND_TRUTH_SITE_C = "ground-truth-site-c"

ANSWERS_SITE_I = "answers-site-i"
EDA_SITE_I = "eda-site-i"
GROUND_TRUTH_SITE_I = "ground-truth-site-i"

# column name constants
SITE = "Site"
RESPONSE_ID = "Response ID"
GROUND_TRUTH = "Ground Truth"

# column name replacements
PARTICIPANT_ID = 'Participant ID'

# sociodemographic column name constants1
AGE = "Please enter your age"
STUDY = "Please specify your program of study."
MOTHER_TONGUE = "Please specify your mother tongue."
GENDER = "Please enter your gender"
HANDEDNESS = "Are you left or right handed?"

def get_merged_distribution_data(selected_site):
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')

    if selected_site == 'magdeburg':
        answers_table = ANSWERS_SITE_M
        eda_table = EDA_SITE_M
        ground_truth_table = GROUND_TRUTH_SITE_M
    elif selected_site == 'chemnitz':
        answers_table = ANSWERS_SITE_C
        eda_table = EDA_SITE_C
        ground_truth_table = GROUND_TRUTH_SITE_C
    elif selected_site == 'ilmenau':
        answers_table = ANSWERS_SITE_I
        eda_table = EDA_SITE_I
        ground_truth_table = GROUND_TRUTH_SITE_I
    
    # ANSWERS
    try:
        df_answers = pd.read_sql_query(f"""
                                            SELECT * 
                                            FROM `{answers_table}`;
                                            """,conn)
    except:
        return "The Answers Table Data for this site is not available. Please upload the data and try again."
    
    if 'P' not in str(df_answers['Response ID'].values[0]):
        df_answers['Response ID'] = 'P' + df_answers['Response ID'].astype(str).str.zfill(2)
        df_answers.reset_index(drop=True)
    # Drop columns containing the substring '[Scrap]'
    columns_to_drop = [col for col in df_answers.columns if '[Scrap]' in col]
    df_answers = df_answers.drop(columns=columns_to_drop, axis=1)
    
    # Create a mapping for column renaming
    column_mapping = {}
    count_mapping = {'Is it a good or scrap cylinder?': 1, 
                    'Please insert the diameter of the cylinder.': 1, 
                    'How certain are you about the answer?': 1}

    for column in df_answers.columns:
        for column_type in count_mapping.keys():
            if column_type in column:
                new_column_name = f'{column_type}_{count_mapping[column_type]}'
                column_mapping[column] = new_column_name
                count_mapping[column_type] += 1

    # Rename columns based on the mapping
    df_answers.rename(columns=column_mapping, inplace=True)

    df_answers_new = pd.DataFrame(columns=['Participant ID', 'Task ID', 'Gender', 'Age', 'Program of Study', 'Mother Tongue', \
                                           'Left/Right Handed', 'Good/Scrap', 'Diameter', 'Certainty', 'Time for each Task', 'Total Time for all 16 Tasks per Participant'])

    # Iterate through rows and then through columns
    for index, row in df_answers.iterrows():
        p_id = row['Response ID']
        gender = row['Please enter your gender']
        age = row['Please enter your age']
        pgm_study = row['Please specify your program of study.']
        language = row['Please specify your mother tongue.']
        left_right = row['Are you left or right handed?']
        total_time = row['Total time']

        # Iterate through task IDs
        for task_id_suffix in range(1, 17):
            task_id = f'T{task_id_suffix:02d}'
            good_scrap = row[f'Is it a good or scrap cylinder?_{task_id_suffix}']
            diameter = row[f'Please insert the diameter of the cylinder._{task_id_suffix}']
            certainty = row[f'How certain are you about the answer?_{task_id_suffix}']
            each_task_time = row[f'Group time: Cylindertask {task_id_suffix}']

            # Append a new row to the reshaped DataFrame
            new_row = pd.Series(pd.Series([p_id, task_id, gender, age, pgm_study, language, left_right, good_scrap, diameter, certainty, each_task_time, total_time],
                                           index=['Participant ID', 'Task ID', 'Gender', 'Age', 'Program of Study', 'Mother Tongue', 'Left/Right Handed', 'Good/Scrap', \
                                                  'Diameter', 'Certainty', 'Time for each Task', 'Total Time for all 16 Tasks per Participant']))
            # Append the new row to the DataFrame
            df_answers_new = pd.concat([df_answers_new, new_row.to_frame().transpose()], ignore_index=True)

    # Replace 'Yes' with 'Good' and 'No' with 'Scrap' in the 'Good/Scrap' column
    df_answers_new['Good/Scrap'] = df_answers_new['Good/Scrap'].replace({'Yes': 'Good', 'No': 'Scrap'})
    columns_to_convert = df_answers_new.columns[2:]
    df_answers_new[columns_to_convert] = df_answers_new[columns_to_convert].applymap(lambda x: x.lower() if isinstance(x, str) else x)
    df_answers_new['Mother Tongue'] = df_answers_new['Mother Tongue'].replace(['deutsch'], 'german', regex=True)
    df_answers_new['Program of Study'] = df_answers_new['Program of Study'].replace(['computer sience', 'informatik', 'phd computer science'], 'computer science', regex=True)
    df_answers_new['Program of Study'] = df_answers_new['Program of Study'].replace(['ingenieurcomputer science'], 'engineering informatics', regex=True)
    df_answers_new['Program of Study'] = df_answers_new['Program of Study'].replace({None: 'missing answer', '': 'missing answer'})

    # EDA
    try:
        df_eda = pd.read_sql_query(f"""
                                        SELECT * 
                                        FROM `{eda_table}`;
                                        """,conn)
    except:
        return "The EDA Table Data for this site is not available. Please upload the data and try again."
    
    if 'P' not in str(df_eda['Response ID'].values[0]):
        df_eda['Response ID'] = 'P' + df_eda['Response ID'].astype(str).str.zfill(2)
        df_eda.reset_index(drop=True)
    
    df_eda_new = pd.DataFrame(columns=['Participant ID', 'Task ID', 'EDA Grand Mean'])
    # Iterate through rows and then through columns
    for index, row in df_eda.iterrows():
        p_id = row['Response ID']
        
        # Iterate through task IDs
        for task_id_suffix in range(1, 17):
            task_id = f'T{task_id_suffix:02d}'
            eda_gm = row[f'T{task_id_suffix:02d}_Eda_GrandMean']
            
            # Append a new row to the reshaped DataFrame
            new_row = pd.Series([p_id, task_id, eda_gm], index=['Participant ID', 'Task ID', 'EDA Grand Mean'])
            df_eda_new = pd.concat([df_eda_new, new_row.to_frame().transpose()], ignore_index=True)
    # GROUND TRUTH
    try:
        df_ground_truth = pd.read_sql_query(f"""
                                                SELECT * 
                                                FROM `{ground_truth_table}`
                                                """,conn)
    except:
        return "The Ground Truth Table Data for this site is not available. Please upload the data and try again."
    
    if 'T' not in str(df_ground_truth['Task ID'].values[0]):
        df_ground_truth['Task ID'] = 'T' + df_ground_truth['Task ID'].astype(str).str.zfill(2)
        df_ground_truth.reset_index(drop=True)
    df_ground_truth['Ground Truth'] = df_ground_truth['Ground Truth'].str.lower()

    # Merge DataFrames based on two common columns
    merged_df = pd.merge(df_answers_new, df_eda_new, on=['Participant ID', 'Task ID'], how='left', suffixes=('', '_eda'))
    final_merged_df = pd.merge(merged_df, df_ground_truth, on=['Task ID'], how='left', suffixes=('', '_ground_truth'))
    final_merged_df = final_merged_df[['Participant ID', 'Task ID', 'Gender', 'Age', 'Program of Study', \
                                       'Mother Tongue', 'Left/Right Handed', 'Good/Scrap', 'Diameter', 'Certainty', \
                                       'Ground Truth', 'Time for each Task', 'Total Time for all 16 Tasks per Participant', 'EDA Grand Mean']]

    conn.close()
    return final_merged_df

# task columns
DIAMETER = "Please insert the diameter of the cylinder."
CERTAINTY = "How certain are you about the answer?"
GROUP_TIME = "Group time"
EDA_GRAND_MEAN = "Eda_GrandMean"

# TODO: do I need 'Task ID','Good/Scrap', 'Total Time for all 16 Tasks per Participant'?
COLUMN_NAME_MAPPING = {RESPONSE_ID: PARTICIPANT_ID, GENDER: 'Gender', AGE: 'Age',STUDY: 'Program of Study', MOTHER_TONGUE: 'Mother Tongue',HANDEDNESS:'Left/Right Handed', DIAMETER: 'Diameter', CERTAINTY: 'Certainty', GROUP_TIME:'Time', EDA_GRAND_MEAN: 'Eda Grand Mean'}

def get_merged_data(get_site_m, get_site_c, get_site_i):
    # Connect to the SQLite database
    conn = sqlite3.connect("database.db")

    df_answers_eda_m = pd.DataFrame()
    df_answers_eda_c = pd.DataFrame()
    df_answers_eda_i = pd.DataFrame()
    
    df_ground_truth_m = pd.DataFrame()
    df_ground_truth_c = pd.DataFrame()
    df_ground_truth_i = pd.DataFrame()
    
    if get_site_m:
        # Execute the SQL query to join the tables
        df_answers_eda_m = pd.read_sql_query(
            f"""
            SELECT * 
            FROM `{ANSWERS_SITE_M}`
            INNER JOIN `{EDA_SITE_M}` USING (`{RESPONSE_ID}`);
            """,
            conn,
        )
        df_answers_eda_m.insert(0, 'Site', 'M')
        
        df_ground_truth_m = pd.read_sql_query(
            f"""
            SELECT * 
            FROM `{GROUND_TRUTH_SITE_M}`
            """,
            conn,
        )
    if get_site_c:
        # Execute the SQL query to join the tables
        df_answers_eda_c = pd.read_sql_query(
            f"""
            SELECT * 
            FROM `{ANSWERS_SITE_C}`
            INNER JOIN `{EDA_SITE_C}` USING (`{RESPONSE_ID}`);
            """,
            conn,
        )
        df_answers_eda_c.insert(0, 'Site', 'C')

        df_ground_truth_c = pd.read_sql_query(
            f"""
            SELECT * 
            FROM `{GROUND_TRUTH_SITE_C}`
            """,
            conn,
        )
    if get_site_i:
        # Execute the SQL query to join the tables
        df_answers_eda_i = pd.read_sql_query(
            f"""
            SELECT * 
            FROM `{ANSWERS_SITE_I}`
            INNER JOIN `{EDA_SITE_I}` USING (`{RESPONSE_ID}`);
            """,
            conn,
        )
        df_answers_eda_i.insert(0, 'Site', 'I')

        df_ground_truth_i = pd.read_sql_query(
            f"""
            SELECT * 
            FROM `{GROUND_TRUTH_SITE_I}`
            """,
            conn,
        )
    
    df_answers_eda = pd.concat([df_answers_eda_m, df_answers_eda_i,df_answers_eda_c], ignore_index=True)
    df_ground_truth = pd.concat([df_ground_truth_m, df_ground_truth_i, df_ground_truth_c],ignore_index=True)
    
    # add task number automatically
    # TODO: put adding of task number to a separate preprocessing function
    new_cols = {}
    tasks = range(1, 17)  # Task numbers from T01 to T16
    num_cols_per_task = [4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    start_index = 8
    for task, num_cols in zip(tasks, num_cols_per_task):
        for i in range(num_cols):
            old_name = df_answers_eda.columns[start_index + i]
            new_name = f"T{str(task).zfill(2)}_{old_name}"
            new_cols[old_name] = new_name
        start_index += num_cols
    df_answers_eda = df_answers_eda.rename(columns=new_cols)
    df_answers_eda.reset_index(drop=True, inplace=True)
    # iterate over answers tables columns
    for column_name in df_answers_eda.columns:
        # insert truth columns after each task response
        is_first_answer = False
        is_answer = False
        if "scrap cylinder?" in column_name:
            if "[Scrap]" in column_name:
                is_first_answer = True
            elif "[Good]" in column_name:
                continue
            else:
                is_answer = True
        if is_first_answer or is_answer:
            # get beginning of column name e.g. T01
            # get the second and third letters
            task_letters = column_name[1:3]  # This will return '01'
            # Try to cast the letters to an integer
            try:
                task_number = int(task_letters)
            except ValueError:
                print("The letters cannot be converted to an integer.")
            ## get truth from ground_truth dataframe
            ground_truth_for_task = df_ground_truth.loc[task_number - 1, GROUND_TRUTH]
            ## insert column called T01 Ground Truth and column T01 Correctness based on comparison of truth to answer
            column_index = df_answers_eda.columns.get_loc(column_name) + 1
            ground_truth_column_name = "T" + task_letters + " Ground Truth"
            df_answers_eda.insert(
                loc=column_index,
                column=ground_truth_column_name,
                value=ground_truth_for_task,
            )
            # determine correctness - iterate over participants and compare ground truth with answers
            correctness_for_task = np.empty(len(df_answers_eda), dtype=object)
            for index, row in df_answers_eda.iterrows():
                answer = row[column_name]
                if is_first_answer:
                    # Answer: Yes --> Scrap
                    if answer == "Yes" and ground_truth_for_task == "Scrap":
                        correctness_for_task[index] = "True"
                    else:
                        correctness_for_task[index] = "False"
                if is_answer:
                    # Answer: Scrap --> Scrap
                    if answer == ground_truth_for_task:
                        correctness_for_task[index] = "True"
                    else:
                        correctness_for_task[index] = "False"
            # insert correctness column:
            column_index = df_answers_eda.columns.get_loc(ground_truth_column_name) + 1
            correctness_truth_column_name = "T" + task_letters + " Correctness"
            df_answers_eda.insert(
                loc=column_index,
                column=correctness_truth_column_name,
                value=correctness_for_task,
            )

    # Close the connection
    conn.close()
    return df_answers_eda


# NOT the ID, date, seed, empty fields (time)
# INCLUDE: correctness, eda,
def get_columns_for_task(df, task_id):
    
    # get sociodemographic columns
    # NOTE: removed HANDEDNESS because it is equal for all participants --> nothing to derive
    column_names = [SITE, RESPONSE_ID, AGE, STUDY, MOTHER_TONGUE, GENDER]
    sociodemographic_columns = df[column_names]
    if task_id == "all":
        # get all task columns
        task_columns = df.filter(regex='^T[0-9]{2}')

        # get all task time columns
        group_time_column = df.filter(regex='^Group.*[0-9]$')
    else:
        # get all task and EDA columns (including group time that ends with Task number e.g. "1")
        task_columns = df.loc[:, df.columns.str.startswith(task_id)]

        # get group time column
        task_letters = task_id[1:3]  # This will return e.g '01'
        # Try to cast the letters to an integer
        try:
            task_number = int(task_letters)
        except ValueError:
            print("The letters cannot be converted to an integer.")
        group_time_column = df.loc[:, df.columns.str.endswith(" " + str(task_number))]
    
    # move task digits to the front for group times
    for col in group_time_column.columns:
        # Find the last two digits in the column name
        last_two_digits = re.search('[0-9]{2}$', col)
        # Check if the last symbol is a number and the second to last is a whitespace
        last_digit_second_to_last_whitespace = re.search('[0-9]$', col)
        # Move the last two digits to the front and remove ": Cylindertask XY"
        if last_two_digits:
            new_col = "T" + last_two_digits.group() + " " + col[:-17]
        elif last_digit_second_to_last_whitespace: 
            new_col = "T0" + last_digit_second_to_last_whitespace.group() + " " + col[:-16]
        else:
            new_col = col
        group_time_column.rename(columns={col: new_col}, inplace=True)
        
    
    # remove: "Txx Is it a good or scrap cylinder?" and "Txx Ground Truth" because the relevant information is the correctness
    column_names = task_columns.filter(
        regex="^(?!.*scrap|.*Ground).*$"
    ).columns.tolist()
    selected_task_columns = task_columns[column_names]
    dataframe = pd.concat([sociodemographic_columns, selected_task_columns], axis=1)

    dataframe = pd.concat([dataframe, group_time_column], axis=1)

    # rename columns for display
    for col in dataframe.columns:
        for key in COLUMN_NAME_MAPPING:
            if key in col:
                updated_col_name = col.replace("_", " ")
                if key in [GROUP_TIME, DIAMETER, CERTAINTY, EDA_GRAND_MEAN]:
                    # Diameter and Certainty have trailing numbers
                    if "." in updated_col_name:
                        # Get the position of the dot
                        dot_pos = updated_col_name.index('.')
                        # Cut off all characters after and including the dot
                        # TODO certainty has a dot at the end
                        updated_col_name = updated_col_name[:dot_pos+1]
                    dataframe.rename(columns={col: updated_col_name.replace(key, COLUMN_NAME_MAPPING[key])}, inplace=True)
                else:
                    dataframe.rename(columns={col: COLUMN_NAME_MAPPING[key]}, inplace=True)
    return dataframe
