import torch.nn.functional as F
import torch

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.hinge = torch.nn.HingeEmbeddingLoss()

    def forward(self, output1, output2, label):      
        dist = torch.nn.functional.l1_loss(output1,output2)     
        loss_contrastive = self.hinge(dist,label)
        return loss_contrastive


"""
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

"""
