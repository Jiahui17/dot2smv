from networkx import (
    has_path,
    find_cycle,
    all_simple_paths,
    DiGraph,
    MultiDiGraph,
)
from src.utils import get_op_type, parse_port, is_operator_or_decider, parse_buffer_attr, parse_constant_value
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

        comp_type = self.nodes[node]["mlir_op"]

        # redeciders = self.get_loop_deciders()

        # input signals
        if is_operator_or_decider(self.nodes[node]):
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

            # if node in redeciders:
            #     comp_type = comp_type.replace("decider", "redecider")
        elif comp_type == "handshake.buffer":
            transparent, slots = parse_buffer_attr(self.nodes[node])
            comp_type = f'_buffer{slots}{"t" if transparent else "o"}'
        elif comp_type == "handshake.constant":  # get constant value in decimal
            const_value = parse_constant_value(self.nodes[node])
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

        sorted_input_channels = sorted([ (pred, eattr) for pred, _, eattr in self.in_edges(node, data=True) ], key=lambda d:int(d[1]["to_idx"]))

        for pred, eattr in sorted_input_channels:

            if comp_type == "handshake.constant":
                dataIn = f"{const_value}"
            else:
                dataIn = f'{pred}.dataOut{parse_port(eattr["from_idx"])}'

            input_signals.append(dataIn)
            input_signals.append(f'{pred}.valid{parse_port(eattr["from_idx"])}')

        sorted_output_channels = sorted([ (succ, eattr) for _, succ, eattr in self.out_edges(node, data=True) ], key=lambda d:int(d[1]["from_idx"]))

        # input signals from successor side
        for succ, eattr in sorted_output_channels:

            input_signals.append(f'{succ}.ready{parse_port(eattr["to_idx"])}')

        input_signals = ", ".join(input_signals)

        return f"VAR {node} : {comp_type}({input_signals});"
