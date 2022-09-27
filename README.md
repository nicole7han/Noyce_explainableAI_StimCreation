# Noyce_explainableAI_StimCreation
 
This repo creates rules and images of categories with features that are randomly sampled from distributions.

# Run scripts in order:

1. create_rules.py
- create a bunch of different rules and define the prototype category feature values
- save "rule_parameters.xlsx"
- save "parameters.json"

2. create_stim.py
- create images according to the rule
- save "stim_info.xlsx"
- save train and test images for each rule

3. create_description.py
- create descriptions according to the rule
- save "stim_info_text.xlsx"

4. paraphrase_description.py (not the best now maybe use other approaches)
- create paraphrase descriptions
- save "stim_info_text_para.xlsx"



# Feature Coding:

## Objects ##
e: ellipse

t: triangle

r: rectangle

c: circle (with a ring)

location of the primary object: (.5, .5) (center of the image by default)


## Features ##
e_f1(ellipse feature 1): width (pix)

e_f2(ellipse feature 2): length (pix)

e_f3(ellipse feature 3):orientation relative to vertical (clockwise +, counterclockwise -)

t_f1(triangle feature 1): height (pix)

t_f2(triangle feature 2): base (pix)

t_f3(triangle feature 3): orientation relative to vertical,  (clockwise +, counterclockwise -)

r_f1(rectangle feature 1): length (pix)

r_f2(rectangle feature 2): width (pix)

r_f3(rectangle feature 3): orientation relative to vertical,  (clockwise +, counterclockwise -)

c_f1(circle feature 1): radius (pix)

c_f2(circle feature 2): contrast

c_f3(circle feature 3): width of ring (pix)
