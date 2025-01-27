import re
from src.utils import is_operator_or_decider


"""
this function finds a list of trivial_edges to check
"""


def find_trival_edges(G):
    trivial_edges = set()
    # we always skip all edges on the path from source unit to a join unit
    for source_unit in G:
        if not G.nodes[source_unit]["mlir_op"] == "handshake.source":
            continue
        queue, visited = [source_unit], []
        while queue != []:
            current_node = queue.pop()
            if current_node not in visited:
                visited.append(current_node)
                for _, child_node in G.out_edges(current_node):
                    num_pred = sum(1 for _ in G.in_edges(child_node))
                    trivial_edges.add((current_node, child_node))
                    if num_pred == 1:
                        queue.append(child_node)
                    elif num_pred == 0:
                        raise ValueError
    for pred, succ in G.edges():
        type_of_pred = G.nodes[pred]["mlir_op"]
        type_of_succ = G.nodes[succ]["mlir_op"]
        # we always skip the edges between MC|LSQ and mc|lsq_load|store_op
        pred_is_mem_access_port = re.match(r"handshake.(load|store)", G.nodes[pred]["mlir_op"])

        succ_is_mem_access_port = re.match(r"handshake.(load|store)", G.nodes[succ]["mlir_op"])

        pred_is_mc_or_lsq = G.nodes[pred]["mlir_op"] in ("handshake.lsq", "handshake.mem_controller")
        succ_is_mc_or_lsq = G.nodes[succ]["mlir_op"] in ("handshake.lsq", "handshake.mem_controller")

        if (pred_is_mem_access_port and succ_is_mc_or_lsq) or (
            pred_is_mc_or_lsq and succ_is_mem_access_port
        ):
            trivial_edges.add((pred, succ))

        # we always skip the edges between MC|LSQ and exit node if the
        # circuit has side effect (i.e., a store operator) or the circuit has an LSQ
        if (pred_is_mc_or_lsq) and G.nodes[succ]["mlir_op"] == "handshake.func":
            trivial_edges.add((pred, succ))

        any_unit_in_the_circuit_has_side_effect = any(
            True for n in G if re.match(r"handshake.store", G.nodes[n]["mlir_op"])
        )

        circuit_has_lsq = any(True for n in G if G.nodes[n]["mlir_op"] == "handshake.lsq")

        # if any_unit_in_the_circuit_has_side_effect or circuit_has_lsq:
        #     if (
        #         type_of_pred.lower() == "operator"
        #         and G.nodes[pred]["op"].lower() == "ret_op"
        #     ):
        #         trivial_edges.add((pred, succ))

    return trivial_edges


# for each edge, we want to check if the data stall event ever happens
def check_valid_not_ready(G):
    gna = lambda n: G.nodes.data()[n]  # get node attribute

    return_buffer = ""
    trivial_edges = find_trival_edges(G)

    # list(G.edges(data=True)) returns a list of tuples (pred, succ, dictofattrs)
    for pred, succ, edgeattr in G.edges(data=True):

        if (pred, succ) in trivial_edges:
            continue

        # elif int(gna(pred)["bbID"]) < 0 and int(gna(succ)["bbID"]) < 0:
        #     continue

        # get the index of the valid signal
        from_ = edgeattr["from_idx"]

        # get the index of the ready signal
        to_ = edgeattr["to_idx"]

        # valid and not ready signal
        valid_not_ready = f"{pred}.valid{from_} -> {succ}.ready{to_}"

        # property
        return_buffer += (
            f"INVARSPEC NAME inv_no_stall_{pred}_nReadyArray_{from_} := ({valid_not_ready});"
            + "\n"
        )

    return return_buffer
