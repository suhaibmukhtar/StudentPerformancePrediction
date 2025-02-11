import sys
import os
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    exc_type,exc_value,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    
    error_message=f"Exception Value:{exc_value}\nError has occured in python file:{file_name}\nLine No.:{line_no}\nError Message:{str(error)}"
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        #inheriting from the Exception class
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)#error message from the function
    
    #inherting one more function 
    def __str__(self):
        logging.info(self.error_message)
        return self.error_message #returing above initialized error-message