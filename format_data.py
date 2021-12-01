from tqdm import tqdm
import re
import pandas as pd
import os
def text_strip(row):
    row = re.sub("(\\t)", " ", str(row)).lower()
    row = re.sub("(\\r)", " ", str(row)).lower()
    row = re.sub("(\\n)", " ", str(row)).lower()

    # Remove _ if it occurs more than one time consecutively
    row = re.sub("(__+)", " ", str(row)).lower()

    # Remove - if it occurs more than one time consecutively
    row = re.sub("(--+)", " ", str(row)).lower()

    # Remove ~ if it occurs more than one time consecutively
    row = re.sub("(~~+)", " ", str(row)).lower()

    # Remove + if it occurs more than one time consecutively
    row = re.sub("(\+\++)", " ", str(row)).lower()

    # Remove . if it occurs more than one time consecutively
    row = re.sub("(\.\.+)", " ", str(row)).lower()

    # Remove the characters - <>()|&©ø"',;?~*!
    row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", " ", str(row)).lower()

    # Remove mailto:
    row = re.sub("(mailto:)", " ", str(row)).lower()

    # Remove \x9* in text
    row = re.sub(r"(\\x9\d)", " ", str(row)).lower()

    # Replace INC nums to INC_NUM
    row = re.sub("([iI][nN][cC]\d+)", "INC_NUM", str(row)).lower()

    # Replace CM# and CHG# to CM_NUM
    row = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", "CM_NUM", str(row)).lower()

    # Remove punctuations at the end of a word
#     row = re.sub("(\.\s+)", " ", str(row)).lower()
    row = re.sub("(\-\s+)", " ", str(row)).lower()
    row = re.sub("(\:\s+)", " ", str(row)).lower()

    # Replace any url to only the domain name
    try:
        url = re.search(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", str(row))
        repl_url = url.group(3)
        row = re.sub(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", repl_url, str(row))
    except:
        pass

    # Remove multiple spaces
    row = re.sub("(\s+)", " ", str(row)).lower()

    # Remove the single character hanging between any two spaces
    row = re.sub("(\s+.\s+)", " ", str(row)).lower()
    return row

def format_data_from_dataset():
    list_of_input = []
    list_of_output = []
    for i in enumerate(tqdm(os.listdir(r'dataset\stories_text_summarization_dataset_train'))):
        with open(r'dataset\stories_text_summarization_dataset_train'+'\\'+i[1],encoding='utf-8') as f:
            data = f.read()
            data = data.split('@highlight')
            list_of_input.append(data[0].replace('\n',' ').replace('  ',' '))
            string = ""
            for j in data[1:]:
                string+=j.replace('\n','')+". "
            list_of_output.append(string)
    
    data = pd.DataFrame(list_of_input,columns=['text'])
    data['ctext'] = list_of_output
    data['text'] = data['text'].apply(text_strip)
    data['ctext'] = data['ctext'].apply(text_strip)
    data.to_csv('data.csv',index=False)