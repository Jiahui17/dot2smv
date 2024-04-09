from networkx import (
    has_path,
    find_cycle,
    all_simple_paths,
    DiGraph,
    MultiDiGraph,
)
from src.utils import get_op_type, parse_port
from src.dfg import DFG
import pygraphviz as pgv
import re

"""
convert the input dataflow IR to a smv netlist
"""


class NetlistWriter(DFG):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write_netlist(self):
        header = f"""
			#include "elastic_components.smv"
			#include "parametrized.smv"
			MODULE main
		"""
        declaration = [header]
        for node in self:
            declaration.append(self.write_instance(node))
        return "\n".join(declaration)

    # write one instantiation of module
    def write_instance(self, node):

        comp_type = self.nodes[node]["type"].lower()

        redeciders = self.get_loop_deciders()

        # input signals
        if comp_type == "operator":
            comp_type = get_op_type(self.nodes[node])
            # HACK: get all the deciders that control whether we take the backedge or
            # not, and for these deciders,
            # use a special implementation. Assume that
            # for each of these decider
            # - they reset with loop repeat (TRUE), i.e. they repeat the iteration at
            # least once
            # - if the current condition is loop repeat (TRUE), then the next condition is
            # decided non-deterministically
            # - if the current condition is loop exit (FALSE), then the next
            # condition is TRUE
            if node in redeciders:
                comp_type = comp_type.replace("decider", "redecider")
        elif comp_type == "buffer":
            transparent = self.nodes[node]["transparent"] == "true"
            slots = self.nodes[node]["slots"]
            comp_type = f'_buffer{slots}{"t" if transparent else "o"}'
        elif comp_type == "constant":  # get constant value in decimal
            const_value = int(self.nodes[node]["value"], 0)
            if const_value == 0:
                const_value = "FALSE"
            else:
                const_value = "TRUE"
        elif comp_type == "delayer":
            latency = self.nodes[node]["latency"]
            comp_type = f"delayer{latency}c"
        elif comp_type in (
            "mc",
            "lsq",
        ):
            comp_type = f'{comp_type}_{self.nodes[node]["memory"]}'

        np = sum(1 for e in self.in_edges(node))

        ns = sum(1 for e in self.out_edges(node))

        comp_type = f"{comp_type}_{np}_{ns}"

        input_signals = []

        # input signals from predecessor side
        for id_, (pred, _, eattr) in self.get_indexed_in_channels(node).items():

            if comp_type == "constant_1_1":
                dataIn = f"{const_value}"
            else:
                dataIn = f'{pred}.dataOut{parse_port(eattr["from"])}'

            input_signals.append(dataIn)
            input_signals.append(f'{pred}.valid{parse_port(eattr["from"])}')

        # input signals from successor side
        for id_, (_, succ, eattr) in self.get_indexed_out_channels(node).items():

            input_signals.append(f'{succ}.ready{parse_port(eattr["to"])}')

        input_signals = ", ".join(input_signals)

        return f"VAR {node} : {comp_type}({input_signals});"
