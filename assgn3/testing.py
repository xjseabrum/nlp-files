from dumb_method import limit_i


# def predict(dataframe, group_id, word_id, word_col, pred_col, dict):
#     for sent in range(max(dataframe[group_id])):
#         max_word = max(dataframe.loc[dataframe[group_id] == sent, word_id])
#         for word in range(1, max_word+1):
#             w = list(dataframe.loc[((dataframe[word_id] == word) & (dataframe[group_id] == sent)), word_col])[0]
#             w_tag = list(dataframe.loc[((dataframe[word_id] == word) & (dataframe[group_id] == sent)), pred_col])[0]
#             if w_tag == "O":
#                 p_tag = b_tag(w, dict)
#                 if p_tag == "B":
#                     dataframe.loc[((dataframe[word_id] == word) & (dataframe[group_id] == sent)), pred_col] = "B" 
#                     num_i = limit_i(word, max_word)
#                     if num_i == 0: pass
#                     dataframe.loc[((dataframe[word_id].isin(np.arange(word+1, word+num_i+1))) & (dataframe[group_id] == sent)), pred_col] = "I" 
#             else: pass



def predict(dataframe, group_id, word_id, word_col, pred_col, dict):
    for sent in range(max(dataframe[group_id])):
        word_list = list(dataframe[dataframe[group_id] == sent, word_col])
        pred_list = ["O"] * len(word_list)
        out_list = [b_tag(x, b_probs) for x in word_list]
        b_idx = [y for y, x in enumerate(out_list) if x == "B"]
        print(b_idx)
        



        max_word = max(dataframe.loc[dataframe[group_id] == sent, word_id])
        for word in range(1, max_word+1):
            w = list(dataframe.loc[((dataframe[word_id] == word) & (dataframe[group_id] == sent)), word_col])[0]
            w_tag = list(dataframe.loc[((dataframe[word_id] == word) & (dataframe[group_id] == sent)), pred_col])[0]
            if w_tag == "O":
                p_tag = b_tag(w, dict)
                if p_tag == "B":
                    dataframe.loc[((dataframe[word_id] == word) & (dataframe[group_id] == sent)), pred_col] = "B" 
                    num_i = limit_i(word, max_word)
                    if num_i == 0: pass
                    dataframe.loc[((dataframe[word_id].isin(np.arange(word+1, word+num_i+1))) & (dataframe[group_id] == sent)), pred_col] = "I" 
            else: pass

