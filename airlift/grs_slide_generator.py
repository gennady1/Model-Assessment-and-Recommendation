# Notes:
# 2025/04/03: Embedded data analysis images of in the PDF slides
# 2025/02/07: created slide generator

import ollama
import os
from datetime import date

# get path to current working folder
current_path = os.getcwd()

today = date.today()
abbriv_date = today.isoformat()         # Output: 2023-03-21
long_date = today.strftime("%B %d, %Y") # Output: February 06, 2025

#example inputs for LLM

marp_first_slide = f"""---
marp: true
title: Analysing simulation data with LLMs
paginate: false
author: staskeg@clarkson.edu
date: {abbriv_date}
theme: enable-all-auto-scaling
auto-scaling: true
footer: Gennady Staskevich / Clarkson University / {abbriv_date}
backgroundImage: url('{current_path}/hero-background.svg')

---

# Extracting Insights from Simulation Data using Large Language Models
![bg left:40% 80%]({current_path}/airlift_challenge_logo.jpg)

#### 
Gennady Staskevich
Clarkson University
{long_date}

"""

table_style = """
<style scoped>
table {
  display: block;
  overflow-x: auto; /* Add horizontal scrolling if content is too wide */
  font-size: clamp(8px, 2vw, 16px); /* Responsive font size */
}
th, td {
  padding: 4px; /* Add some padding for better readability */
  text-align: right;
  border: 3px solid rgba(0, 0, 0, 0.2); /* Add some borders for better readability */
  color: black;

  /* Set border-bottom to make rows distinct */
  border-bottom: 3px solid rgba(0, 0, 0, 0.1);
}

th {
  background-color: rgba(138, 138, 138, 0.8); /* Make the first row's background light grey */
}
</style>
"""


marp_table_slide_example = f"""
---

{table_style}

## Python Dataframe stats

| Metric | Aircraft Quantity | Simulation time (sec) | Total flight distance | Total Lateness | Total waiting to process steps | Missed Deliveries | Score |
|---|---|---|---|---|---|---|---|
| count | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |
| mean | 25.5 | 0.41 | 38.03 | 3383.12 | 144.56 | 44.46 | 461.98 |
| std | 14.58 | 0.09 | 19.32 | 1263.55 | 76.60 | 2.70 | 22.54 |
| min | 1.0 | 0.29 | 1.03 | 1872.0 | 5.0 | 38.0 | 408.79 |
| 25% | 13.25 | 0.33 | 23.99 | 2135.0 | 85.0 | 43.0 | 449.23 |
| 50% | 25.5 | 0.39 | 37.49 | 3174.5 | 149.0 | 45.0 | 466.26 |
| 75% | 37.75 | 0.48 | 56.56 | 4078.25 | 216.75 | 47.0 | 482.93 |
| max | 50.0 | 0.62 | 66.96 | 6681.0 | 260.0 | 48.0 | 492.02 |


---

## Key Insight - Dataframe stats
- The most striking observation is the high variability in total lateness, suggesting that while on average simulations are completing around 3383 seconds late, there are significant outliers pushing this value up.

"""

marp_slide_recommendation = """

---

## Recommendations based on: Correlation Analysis

- **Increase Aircraft Quantity to reduce Missed Deliveries and enhance Scores.
- **Optimize scheduling algorithms.

"""


marp_alg_description = """
---

## Benefits of using Linear Regression `scipy.stats.linregress()` for data analysis

- **Simplicity**: Quickly fits a linear regression model with minimal code, making it easy to use in initial exploratory data analysis.
- **Efficiency**: Provides key statistics such as slope, intercept, r-value (correlation coefficient), p-value, and standard error of the estimate, all in one function call.
- **Versatility**: Can handle both x and y variables being arrays or single values, making it suitable for various simulation data formats.

"""


class SlideGenerator():
    
    POWERPOINT = 'powerpoint'
    PDF = 'pdf'
    HTML = 'html'

    LLM_MODEL = 'mistral-nemo:12b'      # model used

    LLM_TEMPERATURE = .7                # Somewhat conservative (good for coding and correct syntax)
    LLM_TOP_P = .1

    PRINT_LLM_INPUT  = False
    PRINT_LLM_OUTPUT = True

    MARP_HEADER = ""

    marp_markup = ""

    

    # initialize the class
    def __init__(self, slide_type):

        if slide_type == type(str):
            slide_type = slide_type.lower()
            if slide_type == self.POWERPOINT or slide_type == self.HTML or slide_type == self.PDF:
                self.slide_type = slide_type

        else:
            self.slide_type =self.PDF

        # lets add default output folder based on output type
        self.pdf_output_folder = os.path.join(os.getcwd(), self.slide_type)
        if not os.path.exists(self.pdf_output_folder):
            os.makedirs(self.pdf_output_folder)

        # initialize the MARP Header /  first slide
        self.marp_markup = marp_first_slide

        print(f"\n\n-----------------------------------------------------------\n class = slide_generator.py\n Presentation type = {self.slide_type} \n LLM Configuration:\n   Model: {self.LLM_MODEL}\n   Temperature: {self.LLM_TEMPERATURE}\n   Top_P: {self.LLM_TOP_P}\n   Print LLM Input = {self.PRINT_LLM_INPUT}\n   Print LLM Output = {self.PRINT_LLM_OUTPUT}\n------------------------------------------------\n" )


    def save_marp_to_file(self, filename='./mog-airlift-analysis.md'):

        if filename.endswith('.txt'):
            filename = filename.replace('.txt', '.md')

        filename = self.pdf_output_folder + "/" + filename

        with open(filename, 'w') as f:
            f.write(self.marp_markup)


        #create presentation material: PDF, PowerPoint or HTML
        if self.slide_type == self.HTML:
            os.system(f"npx @marp-team/marp-cli@latest --allow-local-files {filename}")
        else:
            os.system(f"npx @marp-team/marp-cli@latest --{self.slide_type} --allow-local-files {filename}")


    # def list_supported_algorithms(self):
    #     return self.supported_algorithms

    # This method will create explanation slide (in MARP format) for algorithm. 
    def add_slide_from_algorithm_description(self, algorithm):

        llm_task = f"""As an expert data scientist, please briefly explain why the {algorithm} is good for analyzing simulation data.
        Please use the following guidance:
        1. Use this title: {algorithm}:
        2. The explanation must be in MARP markdown format and start with ---
        3. Summarize each bulletpoint in few words.
        4. Keep maximum 3 bullet points.


        Here is a good example of MARP formatted slide:
        {marp_alg_description}
        """
        response = ollama.chat(model=self.LLM_MODEL, messages=[
            {
            'role': 'system',
            'content': 'You are an expert data scientist',     #1. set model context 
            },
            {
            'content': llm_task,
            'role': 'user',
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'top_p': self.LLM_TOP_P,
            'temperature': self.LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
        )
        results = response['message']['content']
        # self.marp_markup +=  ('\n\n---\n\n' + results + '\n')
        self.marp_markup +=  ('\n\n' + results + '\n')

  
    def add_recommendation_slide(self, title, image_url, input_data):

        # had to remove from template, otherwise (infrequently) some filenames would also get summarized
        image_path = f":**\n\n![bg left:35% 90%]({current_path}/{image_url})"
        image_placeholder = ":**\n\n"

        llm_task = f"""Please create a recommendation slide from this data:
        {input_data}
        
        Use the following guidance:
        1. Please provide 1 key insight from input data as a single bullet point.
        2. Please provide 1 recommendation from input data as a single bullet point.
        3. The explanation must be in MARP markdown format.
        4. Use this template: 
        **Key Insights - {title}:**

        - Key insight
        - Recommendation

        Lastly, here is a good example of MARP formatted slide:
        {marp_slide_recommendation}
        """
        response = ollama.chat(model=self.LLM_MODEL, messages=[
            {
            'role': 'system',
            'content': 'You are an expert data scientist',     #1. set model context 
            },
            {
            'content': llm_task,
            'role': 'user',
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'top_p': self.LLM_TOP_P,
            'temperature': self.LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
        )
        results = response['message']['content'].replace(image_placeholder, image_path)
        
        # check to make sure that the slide starts with '---'abbriv_date
        if not results.strip().startswith('---'):
            results = '\n\n---\n' + results

        self.marp_markup +=  ('\n' + results + '\n')


    def add_table_slide(self, title, input_data):

        llm_task = f"""I am making presentation slides using MARP from data analysis. Please create a slide that has this title: {title}
        
        Please create a table that summarizes the following data:
        {input_data}
        
        Please use the following guidance:
        1. Please provide 1 key takeaway from data.
        2. Keep each explanation brief.
        3. The explanation must be in MARP markdown format.

        Lastly, here is a good example of MARP formatted slide:
        {marp_table_slide_example}

        """
        response = ollama.chat(model=self.LLM_MODEL, messages=[
            {
            'role': 'system',
            'content': 'You are an expert data scientist',     #1. set model context 
            },
            {
            'content': llm_task,
            'role': 'user',
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'top_p': self.LLM_TOP_P,
            'temperature': self.LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
        )
        results = response['message']['content']

        # print(f"\n\tDEBUG: {results}")
        
        self.marp_markup +=  ('\n' + results + '\n')

    
    def print_mirp_to_console(self):
        print(self.marp_markup)



# ###------- test slide gereator ---------
# supported_algorithms = ["dataframe.describe()", "Correlations analysis using Python's dataframe.iloc[].corr()", "Linear regression analysis using Python's scipy.stats.linregress()", "Variance Inflation Factors (VIF) using Python's statsmodels.stats.outliers_influence.variance_inflation_factor()", "Principal Component Analysis (PCA) using Python's sklearn.decomposition.PCA()"]

# table_data = """
# {
#   "Aircraft Quantity": {
#     "count": 50.0,
#     "mean": 25.5,
#     "std": 14.5773797371,
#     "min": 1.0,
#     "25%": 13.25,
#     "50%": 25.5,
#     "75%": 37.75,
#     "max": 50.0
#   },
#   "Simulation time (sec)": {
#     "count": 50.0,
#     "mean": 0.409186,
#     "std": 0.0924880136,
#     "min": 0.2896,
#     "25%": 0.32535,
#     "50%": 0.3852,
#     "75%": 0.47565,
#     "max": 0.6161
#   },
#   "Total flight distance": {
#     "count": 50.0,
#     "mean": 38.0305583567,
#     "std": 19.3184572788,
#     "min": 1.0274224281,
#     "25%": 23.9856946718,
#     "50%": 37.4947052876,
#     "75%": 56.5550555503,
#     "max": 66.9622705823
#   },
#   "Total Lateness": {
#     "count": 50.0,
#     "mean": 3383.12,
#     "std": 1263.5509671282,
#     "min": 1872.0,
#     "25%": 2135.0,
#     "50%": 3174.5,
#     "75%": 4078.25,
#     "max": 6681.0
#   },
#   "Total waiting to process steps": {
#     "count": 50.0,
#     "mean": 144.56,
#     "std": 76.6034474633,
#     "min": 5.0,
#     "25%": 85.0,
#     "50%": 149.0,
#     "75%": 216.75,
#     "max": 260.0
#   },
#   "Missed Deliveries": {
#     "count": 50.0,
#     "mean": 44.46,
#     "std": 2.6970127087,
#     "min": 38.0,
#     "25%": 43.0,
#     "50%": 45.0,
#     "75%": 47.0,
#     "max": 48.0
#   },
#   "Score": {
#     "count": 50.0,
#     "mean": 461.9830389591,
#     "std": 22.5407423689,
#     "min": 408.7880628088,
#     "25%": 449.2311800344,
#     "50%": 466.2585792405,
#     "75%": 482.9333863121,
#     "max": 492.0184132449
#   }
# }
# """

# final_recommendation_basic_stats = """
# Key insight: The average number of aircraft used in simulations is around 25, with a significant range (1 to 47), indicating varied simulation setups.
# Recommendation: To balance computational efficiency and realism, consider using a range of 13 to 38 aircraft for future simulations.
# """

# sp = SlideGenerator("powerpoint")

# # for alg in supported_algorithms:
# #     sp.add_slide_from_algorithm_description(alg)

# sp.add_table_slide("Info: Python dataframe stats", table_data)
# sp.add_recommendation_slide("dataframe.describe()", "aircraft-quantity-exploratory-data-analysis.png", final_recommendation_basic_stats)

# sp.print_mirp_to_console()
# sp.save_marp_to_file("test.md")

# # os.system(f"npx @marp-team/marp-cli@latest --pdf --allow-local-files test.md")



######
    #      # def add_recommendation_slide(self, title, input_data):
    #     llm_task = f"""Please create a recommendation slide from this data:
    #     {input_data}
        
    #     Use the following guidance:
    #     1. Use this title: Key Insights - `{title}`:
    #     2. Please summarize recommendations into short bullet points.
    #     3. The explanation must be in MARP markdown format.

    #     Lastly, here is a good example of MARP formated slide:
    #     {marp_slide_recommendation}
    #     """
    #     response = ollama.chat(model=self.LLM_MODEL, messages=[
    #         {
    #         'role': 'system',
    #         'content': 'You are an expert data scientist',     #1. set model context 
    #         },
    #         {
    #         'content': llm_task,
    #         'role': 'user',
    #         },
    #     ],
    #     options = {
    #         # 'logit_bias': LLM_LOGIT_BIAS,
    #         'top_p': self.LLM_TOP_P,
    #         'temperature': self.LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
    #     }
    #     )
    #     results = response['message']['content']
        
    #     # check to make sure that the slide starts with '---'abbriv_date
    #     if not results.strip().startswith('---'):
    #         results = '\n\n---\n' + results

    #     self.marp_markup +=  ('\n' + results + '\n')