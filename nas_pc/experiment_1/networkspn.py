from spn.structure.leaves.parametric import *
from spn.structure.Base import Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up

class NetworkSPN:

    def __init__(self, blocks, operations, scope_dict, dim, steps):
        self.blocks = blocks
        self.operations = operations
        self.scope_dict = scope_dict
        self.dim = dim
        self.steps = steps

    def generate_structure(self):
        for i in range(len(self.blocks)):
            sub_spn = self.build_block(self.blocks[i], self.scope_dict[i])
            


    def build_block(self, block, scope):
        single_scopes = []
        somelist = [x for x in block if x < self.dim]
        for x in somelist:
            block.remove(x)
        


    