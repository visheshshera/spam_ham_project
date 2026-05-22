# import sys


# def error_message_detail(error, error_detail: sys):
#     """
#     This function returns detailed error message
#     """
#     _, _, exc_tb = error_detail.exc_info()

#     file_name = exc_tb.tb_frame.f_code.co_filename

#     error_message = "\nError occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
#         file_name, exc_tb.tb_lineno, str(error)
#     )

#     return error_message


# class CustomException(Exception):
#     def __init__(self, error_message, error_detail: sys):
#         super().__init__(error_message)
#         self.error_message = error_message_detail(error_message, error_detail)

#     def __str__(self):
#         return self.error_message


import sys
import traceback


def error_message_detail(error, error_detail: sys):
    """
    Returns detailed error message with traceback information
    """

    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    detailed_traceback = traceback.format_exc()

    error_message = f"""
Error occurred in python script:
File Name   : {file_name}
Line Number : {line_number}
Error Type  : {type(error).__name__}
Error Msg   : {str(error)}

Full Traceback:
{detailed_traceback}
"""

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))

        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
