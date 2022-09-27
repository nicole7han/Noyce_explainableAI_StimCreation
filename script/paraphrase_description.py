from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import numpy as np
import pandas as pd
rules = pd.read_excel('Stim_MultipleRules/rule_parameters.xlsx')
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
n_rules = len(rules)

for i, param in rules.iterrows():

    rule = rules[rules['rule']=='rule{}'.format(i+1)]
    if i!=n_rules-1:
        outpath = 'Stim_MultipleRules/Rules_train/stim_rule{}'.format(i+1)
    else:
        outpath = 'Stim_MultipleRules/Rules_eval/stim_rule{}'.format(i+1)

    stim_info = pd.read_excel('{}/stim_info_text.xlsx'.format(outpath))
    print('creating rule {}'.format(i+1))
    unq_feedbacks = np.unique(stim_info['txt'])
    para_dic = {}

    # create all unique paraphrases
    for f in unq_feedbacks:
        text = "paraphrase: " + f + " </s>"
        encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=30
        )
        paraphrases = []
        for output in outputs:
            line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            paraphrases.append(line)
        para_dic[f] = paraphrases

    texts = []
    row_itr= 1
    for row in stim_info.iterrows():
        # print(row_itr)
        sentence = row[1]['txt']
        paraphrases = para_dic[sentence]

        phrase = np.random.choice(paraphrases, 1)[0]
        texts.append(phrase)
        row_itr +=1

    stim_info['text'] = texts
    stim_info.to_excel('{}/stim_info_text_para.xlsx'.format(outpath),index=None)