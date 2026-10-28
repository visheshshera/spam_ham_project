# src/exception.py
import sys
import traceback


class CustomException(Exception):
    """Wraps any exception with origin file, line, type and full traceback.

    Usage:
        except Exception as e:
            raise CustomException(e, sys) from e
    """

    def __init__(self, error, error_detail=None):
        # error_detail is accepted so existing CustomException(e, sys) calls still work
        super().__init__(str(error))

        # Already wrapped: reuse the original details instead of nesting messages
        if isinstance(error, CustomException):
            self.original_type = error.original_type
            self.original_message = error.original_message
            self.file_name = error.file_name
            self.line_number = error.line_number
            self.traceback_text = error.traceback_text
            return

        tb = getattr(error, "__traceback__", None) or sys.exc_info()[2]
        self.file_name, self.line_number = "Unknown", "Unknown"
        if tb is not None:
            while tb.tb_next is not None:  # walk to where the error was raised
                tb = tb.tb_next
            self.file_name = tb.tb_frame.f_code.co_filename
            self.line_number = tb.tb_lineno

        self.original_type = type(error).__name__
        self.original_message = str(error)
        self.traceback_text = (
            "".join(traceback.format_exception(type(error), error, error.__traceback__))
            if isinstance(error, BaseException)
            else ""
        )

    def to_dict(self) -> dict:
        return {
            "error_type": self.original_type,
            "message": self.original_message,
            "file": self.file_name,
            "line": self.line_number,
        }

    def __str__(self) -> str:
        return (
            f"{self.original_type}: {self.original_message} "
            f"[{self.file_name}:{self.line_number}]"
        )
