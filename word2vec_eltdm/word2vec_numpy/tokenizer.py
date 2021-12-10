import re
from typing import List


class Tokenizer:
    def __init__(self, datapath: str) -> None:
        self.datapath = datapath

    def _get_data(self) -> str:
        with open(self.datapath) as file:
            data = file.read()
        return data

    def get_tokens(self) -> List[str]:
        pattern = re.compile(r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*")

        data = self._get_data().lower()
        # .replace('.', ' <PERIOD> ')\
        # .replace(',', ' <COMMA> ')\
        # .replace('"', ' <QUOTATION_MARK> ')\
        # .replace(';', ' <SEMICOLON> ')\
        # .replace('!', ' <EXCLAMATION_MARK> ')\
        # .replace('?', ' <QUESTION_MARK> ')\
        # .replace('(', ' <LEFT_PAREN> ')\
        # .replace(')', ' <RIGHT_PAREN> ')\
        # .replace('--', ' <HYPHENS> ')\
        # .replace('?', ' <QUESTION_MARK> ')\
        # .replace('\n', ' <NEW_LINE> ')\
        # .replace(':', ' <COLON> ')

        return pattern.findall(data)
