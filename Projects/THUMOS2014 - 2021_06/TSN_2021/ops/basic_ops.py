import torch
import math


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


# class SegmentConsensus(torch.autograd.Function):

#     def __init__(self, consensus_type, dim=1):
#         self.consensus_type = consensus_type
#         self.dim = dim
#         self.shape = None
#     @staticmethod
#     def forward(self, input_tensor):
#         self.shape = input_tensor.size()
#         if self.consensus_type == 'avg':
#             output = input_tensor.mean(dim=self.dim, keepdim=True)
#         elif self.consensus_type == 'identity':
#             output = input_tensor
#         else:
#             output = None

#         return output
#     @staticmethod
#     def backward(self, grad_output):
#         if self.consensus_type == 'avg':
#             grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
#         elif self.consensus_type == 'identity':
#             grad_in = grad_output
#         else:
#             grad_in = None

#         return grad_in
    
#### TEST ####
class SegmentConsensus(torch.autograd.Function):

#     def __init__(self, consensus_type, dim=1):
#         self.consensus_type = consensus_type
#         self.dim = dim
#         self.shape = None
    @staticmethod
    def forward(ctx, input_tensor, consensus_type, dim=1, shape=None):
        shape = input_tensor.size()
        ctx.shape = shape
        ctx.consensus_type = consensus_type
        ctx.dim = dim
        ctx.shape = shape
        if ctx.consensus_type == 'avg':
            output = input_tensor.mean(dim=ctx.dim, keepdim=True)
        elif ctx.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.consensus_type == 'avg':
            grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        elif ctx.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus().apply(input, self.consensus_type,)
