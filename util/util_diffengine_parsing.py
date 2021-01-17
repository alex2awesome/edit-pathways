import re
import pandas as pd 
import difflib
import spacy
import copy

def group_using_diffs(db_diffs):
    id_map_copy = db_diffs[['old_id', 'new_id']].copy()
    old_id2idx_map = id_map_copy.reset_index().set_index('old_id')['index'].to_dict()
    all_lists = []
    to_break = False
    loop_idx = 0
    while len(id_map_copy) > 0 and (not to_break):
        if loop_idx % 1000 == 0:
            print(loop_idx)
        curr_idx = id_map_copy.index[0]
        curr_old_id, curr_new_id = id_map_copy.iloc[0]
        curr_list = [curr_old_id, curr_new_id]
        id_map_copy.drop(curr_idx, inplace=True)
        while curr_new_id != curr_old_id:
            curr_old_id = curr_new_id
            if curr_old_id in old_id2idx_map:
                curr_idx = old_id2idx_map[curr_old_id]
                curr_new_id = id_map_copy.loc[curr_idx]['new_id']
                curr_list.append(curr_new_id)
                id_map_copy.drop(curr_idx, inplace=True)
        all_lists.append(curr_list)
        loop_idx += 1
    return all_lists


class diffHandler():
    def __init__(self, split_words=False, window_size=5):
        self.split_words = split_words
        self.window_size = window_size
        self.nlp = spacy.load('en_core_web_lg', disable=["tagger", "ner"])

    def tokenize(self, a):
        if self.split_words:
            a = a.replace('</p>', '').replace('<p>', '\n\n')
            doc = self.nlp.make_doc(a)
            return [token.text for token in doc]
        else:
            return a

    def output_diff_lists(self, a_1, a_2):
        output = ''
        curr_str = ''
        curr_label = None
        output_list = []
        a_1, a_2 = self.tokenize(a_1), self.tokenize(a_2)

        for item in list(difflib.ndiff(a_1, a_2)):
            label, text = item[0], item[2:]
            if label == '?':
                continue
            if label == curr_label:
                if self.split_words:
                    curr_str += ' ' + text
                else:
                    curr_str += text
            else:
                if curr_label != None:
                    output_list.append({'text': curr_str, 'label': curr_label})

                curr_label = label
                curr_str = text
        output_list.append({'text': curr_str, 'label': curr_label})
        return output_list

    def list_to_html(self, output_list):
        output_html = ''
        label_to_rbg = {
            '+':'rgba(0,255,0,0.3)',
            '-':'rgba(255,0,0,0.3)',
            ' ':'rgba(0,0,0,0.3)'
        }
        get_rbg = lambda x: label_to_rbg[x]    
        for item in output_list:
            text = copy.copy(item['text'])
            text = re.sub('</p>\s*<p>', '<br><br>', text)
            text = re.sub('<p>', '', text)
            text = re.sub('</p>', '', text)
            
            if item['label'] in ['+', '-']:
                html = '<span style="background-color:%s">%s</span>' % (get_rbg(item['label']), text)
            else:
                html = text
            if self.split_words:
                output_html += ' ' + html
            else:
                output_html += html

        if self.split_words:
            ## then, \n\n will be in here
            output_html = '<p>%s</p>' % output_html.strip().replace('\n\n', '</p><p>')

        return output_html


    def rolling_window(self, static_text_list, window_size=None):
        if window_size is None:
            window_size = self.window_size

        text_list = copy.deepcopy(static_text_list)
        output = []
        old_text_list = []
        ## 
        sep = ' ' if self.split_words else ''
        while len(text_list) != len(old_text_list):
            old_text_list = copy.deepcopy(text_list)
            for idx in range(len(text_list) - 2):
                if len(self.tokenize(text_list[idx+1]['text'])) < window_size:
                    ### if it's  ['+', ' ', '-'] or ['-', ' ', '+']
                    if (
                        (text_list[idx+1]['label'] == ' ') and (
                            (text_list[idx]['label'] == '+' and text_list[idx+2]['label'] == '-') or 
                            (text_list[idx]['label'] == '-' and text_list[idx+2]['label'] == '+')
                        )
                    ):
                        text_list[idx]['text'] += sep + text_list[idx + 1]['text']
                        text_list[idx + 2]['text'] = text_list[idx + 1]['text'] + sep + text_list[idx + 2]['text']
                        text_list[idx + 1]['text'] = ''

                    ### if it's ['-', ' ', '-'] or ['+', ' ', '+']
                    elif ((text_list[idx + 1]['label'] == ' ') and 
                          (text_list[idx]['label'] in ['+', '-']) and 
                          (text_list[idx]['label'] == text_list[idx+2]['label'])
                     ):
                        ### if it's ('+')['-', ' ', '-'] or ('-')['+', ' ', '+']
                        if (
                              (idx > 0) and 
                              (text_list[idx - 1]['label'] in ['-', '+']) and 
                              (text_list[idx - 1]['label'] != text_list[idx]['label'])
                        ):
                            text_list[idx - 1]['text'] += sep + text_list[idx + 1]['text']
                            text_list[idx]['text'] += sep + text_list[idx + 1]['text']
                            text_list[idx + 1]['text'] = ''
            
                ### if it's  ['+', '-', '+'] or ['-', '+', '-']
                if ((text_list[idx+1]['label'] in ['+', '-']) and (
                        (text_list[idx]['label'] == '+' and text_list[idx+2]['label'] == '+') or 
                        (text_list[idx]['label'] == '-' and text_list[idx+2]['label'] == '-')
                    )
                ):
                    text_list[idx]['text'] += sep + text_list[idx + 2]['text']
                    text_list[idx + 2]['text'] = ''
            
            for idx in range(len(text_list) - 1):
                ### if it's ['-', '-'] or ['+', '+'] or [' ', ' ']
                if (text_list[idx]['label'] == text_list[idx + 1]['label']):
                    text_list[idx]['text'] += sep + text_list[idx + 1]['text']
                    text_list[idx + 1]['text'] = ''

            text_list = list(filter(lambda x: x['text'] != '', text_list))
        
        # second pass
        old_text_list = []
        while len(text_list) != len(old_text_list):
            old_text_list = copy.deepcopy(text_list)
            for idx in range(len(text_list)-2):
                if len(self.tokenize(text_list[idx+1]['text'])) < window_size:
                    ### if it's ['-', ' ', '-'] or ['+', ' ', '+']
                    if ((text_list[idx + 1]['label'] == ' ') and 
                          (text_list[idx]['label'] in ['+', '-']) and 
                          (text_list[idx]['label'] == text_list[idx+2]['label'])
                     ):
                        ### if it's ['-', ' ', '-']('+') or ['+', ' ', '+']('-')
                        if (
                            (idx + 3 < len(text_list)) and 
                            (text_list[idx + 3]['label'] in ['-', '+']) and 
                            (text_list[idx + 3]['label'] != text_list[idx]['label'])
                        ):
                            text_list[idx + 3]['text'] = text_list[idx + 1]['text'] + sep + text_list[idx + 3]['text']
                            text_list[idx]['text'] += sep + text_list[idx + 1]['text']
                            text_list[idx + 1]['text'] = ''

            for idx in range(len(text_list) - 1):
                ### if it's ['-', '-'] or ['+', '+'] or [' ', ' ']
                if (text_list[idx]['label'] == text_list[idx + 1]['label']):
                    text_list[idx]['text'] += sep + text_list[idx + 1]['text']
                    text_list[idx + 1]['text'] = ''

                                
            text_list = list(filter(lambda x: x['text'] != '', text_list))
            
        # final pass
        old_text_list = []
        while len(text_list) != len(old_text_list):
            old_text_list = copy.deepcopy(text_list)
            for idx in range(len(text_list)-2):
                if len(self.tokenize(text_list[idx+1]['text'])) < window_size:
                    ### if it's ['-', ' ', '-'] or ['+', ' ', '+']
                    if ((text_list[idx + 1]['label'] == ' ') and 
                          (text_list[idx]['label'] in ['+', '-']) and 
                          (text_list[idx]['label'] == text_list[idx+2]['label'])
                     ):
                        ### if it's (' ')['-/+', ' ', '-/+'](' ')
                        if (
                            ((idx > 0) and 
                             ((idx + 3) < len(text_list)) and 
                             (text_list[idx - 1]['label'] == ' ') and 
                             (text_list[idx + 3]['label'] == ' ')
                            )
                            or ((idx == 0) and (text_list[idx + 3] == ' '))
                            or (((idx + 3) >= len(text_list)) and (text_list[idx - 1] == ' '))
                        ):
                            text_list[idx]['text'] += sep + text_list[idx + 1]['text']
                            text_list[idx + 1]['text'] = ''

            for idx in range(len(text_list) - 1):
                ### if it's ['-', '-'] or ['+', '+'] or [' ', ' ']
                if (text_list[idx]['label'] == text_list[idx + 1]['label']):
                    text_list[idx]['text'] += sep + text_list[idx + 1]['text']
                    text_list[idx + 1]['text'] = ''

                                
            text_list = list(filter(lambda x: x['text'] != '', text_list))

        
        return text_list