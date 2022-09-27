import os, cv2, json
import pandas as pd
from collections import OrderedDict
from scipy.stats import norm, multivariate_normal
rules = pd.read_excel('Stim_MultipleRules/rule_parameters.xlsx')
file = open('Stim_MultipleRules/parameters.json')
parameters = json.load(file)
objectname = {'c':'ring', 't':'triangle', 'r':'rectangle', 'e':'ellipse'}
objecttxt = {'e':OrderedDict({'width':['smaller', 'larger'],
                              'length':['shorter', 'longer'],
                              'oriented':['more vertically', 'more horizontally'] }),
             'c':OrderedDict({'radius':['smaller', 'larger'],
                              'contrast':['lower', 'higher'],
                              'width':['smaller', 'larger'] }),
             't': OrderedDict({'height': ['smaller', 'larger'],
                               'base': ['smaller', 'larger'],
                               'oriented': ['more vertically', 'more horizontally']}),
             'r': OrderedDict({'length': ['smaller', 'larger'],
                               'width': ['smaller', 'larger'],
                               'oriented': ['more vertically', 'more horizontally']}),
             }

n_rules = len(rules)
for i, param in rules.iterrows():
    rule = rules[rules['rule']=='rule{}'.format(i+1)]
    objects = [rule.obj1.item(), rule.obj2.item(), rule.obj3.item(), rule.obj4.item()]
    if i!=n_rules-1:
        outpath = 'Stim_MultipleRules/Rules_train/stim_rule{}'.format(i+1)
    else:
        outpath = 'Stim_MultipleRules/Rules_eval/stim_rule{}'.format(i+1)
        
    stim_info = pd.read_excel('{}/stim_info.xlsx'.format(outpath))
    print('creating descriptions for rule {}'.format(i+1))

    for j, feat in stim_info.iterrows():
        txt = 'This is Category A. ' if 'A' in feat.stim else 'This is Category B. '
        own_cat = 'A' if 'A' in feat.stim else 'B'
        other_cat = 'B' if 'A' in feat.stim else 'A'
        txt += 'Most Category {} images contain '.format(own_cat)
        primaryobj =  objectname[objects[0]]
        for k in range(4): # for each object

            if k==0: # primary object
                obj = objects[k]
                feat_means = rule['{}_f1_{}'.format(obj,other_cat)].item(), rule['{}_f2_{}'.format(obj,other_cat)].item(), rule['{}_f3_{}'.format(obj,other_cat)].item()

                objn = objectname[obj]
                idx = 0
                for f, val in objecttxt[obj].items(): # loop through each feature
                    img_val, other_val = feat['{}_f{}'.format(obj,idx+1)], feat_means[idx]
                    # print('own value {} vs {}'.format(img_val, other_val))
                    if f=='oriented': # for orientation, we need to use the absolute value to compare which one is more orientented vertically
                        p_d = norm.cdf(abs(img_val), loc=abs(other_val), scale=parameters['{}cov'.format(obj)][idx][
                            idx])
                    else:
                        p_d = norm.cdf(img_val, loc=other_val, scale=parameters['{}cov'.format(obj)][idx][
                            idx])  # cdf of feature value given other category distribution
                    if p_d <= .3:  # most current category is smaller than the other category
                        if idx ==0:
                            txt += "{} with {} {}, ".format(objn, val[0],f)
                        elif idx ==1:
                            txt += "{} {}, ".format(val[0], f)
                        else:
                            txt += "{} {}; ".format(val[0], f)
                    elif p_d >= .7:  # most current category is larger
                        if idx == 0:
                            txt += "{} with {} {}, ".format(objn, val[1],f)
                        elif idx == 1:
                            txt += "{} {}, ".format(val[1], f)
                        else:
                            txt += "{} {}; ".format(val[1], f)
                    else:  # around 50% target is
                        if idx == 0:
                            txt += "{} with similar {}, ".format(objn, f)
                        elif idx == 1:
                            txt += "similar {}, ".format(f)
                        else:
                            txt += "similar {}; ".format(f)
                    idx +=1

            elif k == 1:  # secondary object
                obj = objects[k]
                feat_ref = rule['obj2_feat'].item()
                feat_means = rule['{}_f1_{}'.format(obj,other_cat)].item(), rule['{}_f2_{}'.format(obj,other_cat)].item(), rule['{}_f3_{}'.format(obj,other_cat)].item()
                objn = objectname[obj]
                idx = 0
                for f, val in objecttxt[obj].items():
                    if f in feat_ref: # find the relevant feature
                        break
                    idx += 1
                img_val, other_val = feat['{}_f{}'.format(obj, idx + 1)], feat_means[idx]
                # print('own value {} vs {}'.format(img_val, other_val))
                if f == 'oriented':  # for orientation, we need to use the absolute value to compare which one is more orientented vertically
                    p_d = norm.cdf(abs(img_val), loc=abs(other_val), scale=parameters['{}cov'.format(obj)][idx][
                        idx])
                else:
                    p_d = norm.cdf(img_val, loc=other_val, scale=parameters['{}cov'.format(obj)][idx][
                        idx])  # cdf of feature value given other category distribution
                if p_d <= .3:  # most current category is smaller
                    txt += "The {} has {} {}; ".format(objn, val[0], f)
                elif p_d >= .7:  # most current category is larger
                    txt += "The {} has {} {}; ".format(objn, val[1], f)
                else:  # around 50% target is
                    txt += "The {} has similar {}; ".format(objn, f)

            elif k == 2:  # distance object
                obj = objects[k]
                img_val, other_val = feat['obj3_dis'], rule['obj3_dis_{}'.format(other_cat)].item()
                objn = objectname[obj]
                # print('own value {} vs {}'.format(img_val, other_val))
                p_d = norm.cdf(img_val, loc=other_val,
                               scale=parameters['dis_sd'])  # cdf of lth given distractor distribution
                if p_d <= .3:  # most current category is closer
                    txt += "The {} is closer to the {}. ".format(objn, primaryobj)
                elif p_d >= .7:  # most current category is father
                    txt += "The {} is farther away from the {}. ".format(objn, primaryobj)
                else:  # around 50% target is
                    txt += "The {} has similar distance away from the {}. ".format(objn, primaryobj)
        stim_info.loc[j,'txt']=txt
    stim_info.to_excel('{}/stim_info_text.xlsx'.format(outpath), index=None)