from constants import usecase_constant
from page.general import general_page
from page.dataset_analyser import dataset_analyser_page

def page(usecase,domain_name):
    if usecase==usecase_constant.CHATBOT:
        general_page.general_search(domain_name)
    elif usecase==usecase_constant.DATASET_ANALYSER:
        dataset_analyser_page.dataset_analysis(domain_name)