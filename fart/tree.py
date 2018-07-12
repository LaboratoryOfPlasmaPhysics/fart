


class Node():
    """A standard Node object, with four children"""
    def __init__(self, parent, rect):

        self.parent = parent
        self.children = [None, None, None, None]

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

        self.rect = rect
        self.x0 = rect[0]
        self.y0 = rect[1]
        self.x1 = rect[2]
        self.y1 = rect[3]
        self.x_half = 0.5*(self.x1-self.x0)
        self.y_half = 0.5*(self.y1-self.y0)
        self.x_length = self.x1-self.x0
        self.y_length = self.y1-self.y0




    def lower_left(self):
        return (self.x0,
                self.y0,
                self.x0 + 0.5 * self.x_length,
                self.y0 + 0.5 * self.y_length)

    def lower_right(self):
        return (self.x0 + 0.5 * self.x_length,
                self.y0,
                self.x1,
                self.y0 + 0.5 * self.y_length)

    def upper_left(self):
        return (self.x0,
                self.y0 + 0.5 * self.y_length,
                self.x0 + 0.5 * self.x_length,
                self.y1)

    def upper_right(self):
        return (self.x0 + 0.5 * self.x_length,
                self.y0 + 0.5 * self.y_length,
                self.x1,
                self.y1)


    def subdivide(self):
        logic_need_sub = self.needs_subdivision(self.rect)

        if not logic_need_sub :
            return

        rects = []
        rects.append(self.lower_left())
        rects.append(self.lower_right())
        rects.append(self.upper_left())
        rects.append(self.upper_right())

        for i_r, rect in enumerate(rects):
            self.children[i_r] = self.getinstance(rect)

        self.children[0].subdivide() # << recursion
        self.children[1].subdivide() # << recursion
        self.children[2].subdivide() # << recursion
        self.children[3].subdivide() # << recursion



    def getinstance(self,rect):
        return Node(self,rect)


    def needs_subdivision(self, rect):
        raise NotImplementedError("implement")


    def has_converged(self):
        raise NotImplementedError("implement")






class QuadTree:

    def __init__(self, root_node, max_depth):
        self.leaves = []
        self.allnodes = []
        self.max_depth = max_depth
        root_node.subdivide() # constructs the network of nodes
        self.prune(root_node)
        self.traverse(root_node)


    def prune(self, node):

        if all(v is None for v in node.children):
            return 1

        leafcount = 0
        removals = []

        for child in node.children:
            if child is not None:
                leafcount += self.prune(child)  # <-- recursion

                if leafcount == 0:
                    # no leaves on that part of the tree
                    # tag for removal
                    removals.append(child)

        # remove all dead branches
        for item in removals:
            n = node.children.index(item)
            node.children[n] = None
        return leafcount



    def traverse(self, node):
        self.allnodes.append(node)

        if node.has_converged():
            self.leaves.append(node)

        for child in node.children:
            if child is not None:
                self.traverse(child) # << recursion
