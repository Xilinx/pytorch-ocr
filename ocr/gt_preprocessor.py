import ocr.utils as utils
import torch 

class GtPreprocessor(object):
    def __init__(self):
        self.token2char = {}
        self.char2token = {}
        self.current_token = 0

    @property
    def chars_to_remove(self):
        return map(str, ['\n', '\t', '$']) #$ is in the val dataset but not the train one, remove

    @property
    def chars(self):
        return self.char2token.keys()

    @property
    def number_of_tokens(self):
        return self.current_token + 1 #accounts for the blank token 

    @property
    def blank_token(self):
        return 0 #Warp-CTC defines 0 as the blank token

    def register_unencoded_gt(self, unencoded_gt):
        for c in unencoded_gt:
            if c not in self.char2token and c not in self.chars_to_remove:
                self.current_token += 1
                self.token2char[self.current_token] = c
                self.char2token[c] = self.current_token

    def encode_unencoded_gt(self, unencoded_gt):
        result = []
        for c in unencoded_gt:
            if c not in self.chars_to_remove:
                result.append(self.char2token[c])
        return result
    

