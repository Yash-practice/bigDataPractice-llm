from constants import domain_constant, usecase_constant
from page.general import general_page
from page.social_media import social_media_page

def page(usecase,domain_name):
    if usecase==usecase_constant.CHATBOT:
        general_page.general_search(domain_name)
    elif usecase==usecase_constant.DATASET_ANALYSER:
        social_media_page.social_media_analysis(domain_name)