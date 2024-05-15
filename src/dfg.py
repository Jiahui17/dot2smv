from src.utils import parse_port
from networkx import dfs_preorder_nodes
from math import ceil
from networkx import (
    MultiDiGraph,
    DiGraph,
    Graph,
    nx_agraph,
    simple_cycles,
)
import re


class CFG(MultiDiGraph):
    pass


class DFG(MultiDiGraph):
    """get a dictionary of input channels (pred, succ, attr), indexed by the id_"""

    def get_indexed_in_channels(self, n):
        # the keys of the returned dictionary is a sorted list
        return dict(
            sorted(
                [
                    (int(parse_port(ch[2]["to"])), ch)
                    for ch in self.in_edges(n, data=True)
                ],
                key=lambda d: d[0],
            )
        )

    """ get a dictionary of output channels (pred, succ, attr), indexed by the id_ """

    def get_indexed_out_channels(self, n):
        # the keys of the returned dictionary is a sorted list
        return dict(
            sorted(
                [
                    (int(parse_port(ch[2]["from"])), ch)
                    for ch in self.out_edges(n, data=True)
                ],
                key=lambda d: d[0],
            )
        )

    """
    this class method takes an edge as input, and finds the original
    predecessor and successor in the non-buffered circuit
    """

    def get_original_edge(self, edge: tuple) -> (str, str, dict):
        if edge not in self.edges:
            raise ValueError

        # Types that are inserted by either FPGA'20/FPL'22 buffer, our my
        # ad-hoc buffer (i.e., not in the original program)
        _SKIPPED_TYPES = (
            "Buffer",
            "Dummy",
            "OEHB",
            "TEHB",
            "Delayer",
        )
        pred = edge[0]
        succ = edge[1]
        from_ = self.edges[pred, succ, 0]["from"]
        to_ = self.edges[pred, succ, 0]["to"]
        while self.nodes[pred]["type"] in _SKIPPED_TYPES:
            in_edges = list(self.in_edges(pred))
            if len(in_edges) > 1:
                raise ValueError(
                    f"buffer unit {pred} has more than one input edge, which is impossible!"
                )
            from_ = self.edges[in_edges[0][0], pred, 0]["from"]
            pred = in_edges[0][0]

        while self.nodes[succ]["type"] in _SKIPPED_TYPES:
            out_edges = list(self.out_edges(succ))
            if len(out_edges) > 1:
                raise ValueError(
                    f"buffer unit {pred} has more than one output edge, which is impossible!"
                )
            to_ = self.edges[succ, out_edges[0][1], 0]["to"]
            succ = out_edges[0][1]
        return (pred, succ, {"from": from_, "to": "to_"})

    """
    this function returns a set of cfg edges {(bb0, bb1), (bb1, bb2)}
    it is useful for checking if the graph is refered by a bb cycle
    """

    def get_cfg_edges_in_dfg(self):
        set_of_cfg_edges = set()
        for branch, v in self.edges():
            # get the original edge
            _, next_, _ = self.get_original_edge((branch, v))
            bb_prev = int(self.nodes[branch]["bbID"])
            bb_next = int(self.nodes[next_]["bbID"])
            if not (bb_prev == 0 or bb_next == 0):
                set_of_cfg_edges.add((bb_prev, bb_next))
        return set_of_cfg_edges

    """
        this class method returns a list of deciders
        that determines the loop exit/repeat of each loops
    """

    def get_loop_deciders(self):
        return get_loop_deciders(self)

    def to_dot(self, name: str = "test.dot"):
        nx_agraph.to_agraph(self).write(f"./reports/{name}")


"""
    this function returns a list of deciders
    that determines the loop exit/repeat of each loops
"""


def get_loop_deciders(dfg):

    # get all BBs that have a back edge
    cfg = CFG()
    cfg.add_edges_from(dfg.get_cfg_edges_in_dfg())

    cfg_order = list(
        dfs_preorder_nodes(
            cfg, source=min(list(n for n in cfg), key=lambda n: int(n))
        )
    )

    deciders_to_return = set()

    for bb, bb_next in cfg.edges():
        # an CFG edge is a back edge if the edge U -> V has V < U in dfs
        # visit order.
        if cfg_order.index(bb) < cfg_order.index(bb_next):
            continue

        for decider in (
            n
            for n in dfg
            if int(dfg.nodes[n]["bbID"]) == bb
            and dfg.nodes[n]["type"] == "Operator"
            and re.match(f"(f|i)cmp_\w+_op", dfg.nodes[n]["op"])
        ):

            # HACK: at least this unit is a decider, and it is inside the
            # BB that has a back edge, but is it enough?

            _feed_by_constant = False
            # for the decider in BB, check if it is feed by a constant
            queue, visited = [decider], []
            while queue != []:
                current_node = queue.pop()
                if current_node not in visited:
                    visited.append(current_node)
                    for parent_node, _ in dfg.in_edges(current_node):
                        if dfg.nodes[parent_node]["type"] in (
                            "Merge",
                            "CntrlMerge",
                            "Mux",
                        ):
                            continue
                        elif dfg.nodes[parent_node]["type"] == "Constant":
                            _feed_by_constant = True
                            continue
                        else:
                            queue.append(parent_node)
            # NOTE: check if the output is feeding a branch directly, if
            # this is feeding a select, then we should not regulate its
            # condition generation
            _feeding_a_branch = False
            queue, visited = [decider], []
            while queue != []:
                current_node = queue.pop()
                if current_node not in visited:
                    visited.append(current_node)
                    for _, child_node in dfg.out_edges(current_node):
                        # NOTE: I think there are only three types
                        # of units that accept a condition as input
                        child_node_is_select = (
                            dfg.nodes[child_node]["type"] == "Operator"
                            and dfg.nodes[child_node]["op"] == "select_op"
                        )
                        child_node_is_branch = (
                            dfg.nodes[child_node]["type"] == "Branch"
                        )
                        child_node_is_mux = (
                            dfg.nodes[child_node]["type"] == "Mux"
                        )
                        if child_node_is_select:
                            # if the child node is a select then
                            # the decider condition generation is
                            # not regulated
                            continue
                        elif child_node_is_branch:
                            # if it is feeding a branch, then yes
                            _feeding_a_branch = True
                            continue
                        elif child_node_is_mux:
                            # if it is feeding a Mux, then something must be wrong
                            assert (
                                False
                            ), f"It appears that the {decider} is directly driving a Mux, which seems impossible"
                        else:
                            queue.append(child_node)
            if _feed_by_constant and _feeding_a_branch:
                deciders_to_return.add(decider)

    return deciders_to_return
