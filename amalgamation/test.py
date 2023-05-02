#%%
class node:
    def __init__(self,val) -> None:
        self.val = val
        self.next_nodes = []

checked_set = set()

def find_all_nodes(node):
    all_nodes = [node]
    cur_nodes = node.next_nodes
    for node in cur_nodes:
        checked_set.add(node)
    while len(cur_nodes):
        all_nodes += cur_nodes
        new_nodes = []
        for node in cur_nodes:
            if node not in checked_set:
                new_nodes += node.next_nodes
            checked_set.add(node)
        cur_nodes = new_nodes
    return all_nodes