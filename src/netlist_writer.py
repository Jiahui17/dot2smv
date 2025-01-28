from src.utils import (
    get_op_type,
    is_operator_or_decider,
    parse_buffer_attr,
    parse_constant_value,
    Dot2SmvNotImplementedError,
)
from src.dfg import DFG
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

        np = sum(1 for e in self.in_edges(node))

        ns = sum(1 for e in self.out_edges(node))

        # input signals
        if is_operator_or_decider(self.nodes[node]):
            comp_type = get_op_type(self.nodes[node])
        elif comp_type == "handshake.buffer":
            transparent, slots = parse_buffer_attr(self.nodes[node])
            comp_type = f'_buffer{slots}{"t" if transparent == "true" else "o"}'
        elif comp_type == "handshake.constant":  # get constant value in decimal
            const_value = parse_constant_value(self.nodes[node])
            comp_type = comp_type.replace("handshake.", "")
        elif comp_type == "delayer":
            raise Dot2SmvNotImplementedError(
                "Memory access is not yet supported!", self.nodes[node]
            )
            latency = self.nodes[node]["latency"]
            comp_type = f"delayer{latency}c"
        elif comp_type in ("handshake.mem_controller", "handshake.lsq"):
            memory = re.search(
                r"(MC|LSQ) \((\w+)\)", self.nodes[node]["label"]
            ).group(2)
            comp_type = comp_type.replace("handshake.", "") + f"_{memory}"
            raise Dot2SmvNotImplementedError(
                "Memory access is not yet supported!", self.nodes[node]
            )
        elif comp_type == "handshake.func" and np == 0:
            comp_type = "entry"
        elif comp_type == "handshake.func" and ns == 0:
            comp_type = "sink"
        elif "handshake.load" in comp_type or "handshake.store" in comp_type:
            raise Dot2SmvNotImplementedError(
                "Memory access is not yet supported!", self.nodes[node]
            )
        elif comp_type == "handshake.ndwire":
            comp_type = "ndw"
        elif "handshake" in comp_type:
            comp_type = comp_type.replace("handshake.", "")

        input_signals = []

        # input signals from predecessor side

        sorted_input_channels = sorted(
            [
                (pred, eattr)
                for pred, _, eattr in self.in_edges(node, data=True)
            ],
            key=lambda d: int(d[1]["to_idx"]),
        )

        for pred, eattr in sorted_input_channels:

            # HACK: The from_idx of function argument is not enumerated
            # according to the output id of the DOT node
            from_idx = eattr["from_idx"] if self.nodes[pred]["mlir_op"] != "handshake.func" else 0

            if comp_type == "constant":
                dataIn = f"{const_value}"
            else:
                dataIn = f'{pred}.dataOut{from_idx}'

            input_signals.extend([dataIn, f'{pred}.valid{from_idx}'])

        sorted_output_channels = sorted(
            [
                (succ, eattr)
                for _, succ, eattr in self.out_edges(node, data=True)
            ],
            key=lambda d: int(d[1]["from_idx"]),
        )

        # input signals from successor side
        for succ, eattr in sorted_output_channels:
            to_idx = eattr["to_idx"] if self.nodes[succ]["mlir_op"] != "handshake.func" else 0

            input_signals.append(f'{succ}.ready{to_idx}')

        input_signals = ", ".join(input_signals)
        comp_type = f"{comp_type}_{np}_{ns}"

        return f"VAR {node} : {comp_type}({input_signals});"
