from constants import usecase_constant
from page.general import general_page
from page.dataset_analyser import dataset_analyser_page
from page.audio_analysis import audio_page
from page.email_analysis import email_analysis

def page(usecase,domain_name):
    if usecase==usecase_constant.CHATBOT:
        general_page.general_search(domain_name)
    elif usecase==usecase_constant.DATASET_ANALYSER:
        dataset_analyser_page.dataset_analysis(domain_name)
    elif usecase==usecase_constant.AUDIO_ANALYSER:
        audio_page.main(domain_name)
    elif usecase==usecase_constant.EMAIL_ANALYSER:
        email_analysis.main(domain_name)
    
