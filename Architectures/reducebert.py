import sys
import torch
from torch import nn
from transformers import BertModel

sys.path.append("../")
from consts import *



class ReduceBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.reduce_positive_dim = nn.Sequential(nn.Linear(BERT_DIM, REVIEW_DIM))
        self.reduce_negative_dim = nn.Sequential(nn.Linear(BERT_DIM, REVIEW_DIM))
        self.reduce_review_dim = nn.Sequential(nn.Linear(REVIEW_DIM * 2, REVIEW_DIM))

    def forward(self, review):
        review_encoded_positive, review_encoded_negative = review

        positive_vector = self.bert(**review_encoded_positive)["pooler_output"]
        negative_vector = self.bert(**review_encoded_negative)["pooler_output"]

        positive_vector = self.reduce_positive_dim(positive_vector)
        negative_vector = self.reduce_negative_dim(negative_vector)

        review_vector = torch.cat((positive_vector, negative_vector), dim=1)
        review_vector = self.reduce_review_dim(review_vector)
        return review_vector
