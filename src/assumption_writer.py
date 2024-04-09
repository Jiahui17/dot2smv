from src.dfg import DFG
import pygraphviz as pgv
import networkx as nx


class AssumptionWriter(DFG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # write invariant constraints to make the behavior of smv model more realistic
    def write_assumptions(self):
        def gna(n):
            return self.nodes.data()[
                n
            ]  # get_node_attribute: return the dictionary of the node attributes

        return_buffer = f"""
		"""
        # all nodes connected to MC at maximum is stalled by ndw by
        # number of steps upper bounded by the number of nodes connected to MC

        for mc in filter(lambda n: gna(n)["type"] == "MC", self.nodes()):
            # get the number of nodes connected to MC
            n_mem_nodes = int(self.nodes.data()[mc]["ldcount"]) + int(
                self.nodes.data()[mc]["stcount"]
            )
            # TODO: what happend if we apply compositional checking?
            assert n_mem_nodes > 0, (
                "error - MC node %s has zero memory port attached to it!" % mc
            )

            load_nodes = list(
                filter(
                    lambda n: "op" in gna(n) and gna(n)["op"] == "mc_load_op",
                    self.predecessors(mc),
                )
            )
            # NOTE: the events of load request not granted
            load_request = list()
            load_grant = list()

            for mem_node in load_nodes:
                wire_state = f"{mem_node}.ndw0.state"
                if len(load_nodes) == 1:
                    return_buffer += f"INVAR (({wire_state}) = running);\n"

                load_request.append(f"({mem_node}.b0.valid0)")
                load_grant.append(f"(({wire_state} = running))")

            for id_ in range(len(load_request)):
                # constraints: when the load is the only one that requests for it, that it can get it
                is_only_request = " & ".join(
                    [f"(!{m})" for m in load_request[:id_] + load_request[id_ + 1 :]]
                    + [load_request[id_]]
                )
                constr = f"({is_only_request} -> {load_grant[id_]})"
                return_buffer += f"INVAR {constr};\n"
                # TODO: when there are multiple load requests, one of
                # them must be granted for access when there are
                # multiple load units, we say we arbitrate them in
                # random ways.

            store_request = list()
            store_grant = list()

            store_nodes = list(
                filter(
                    lambda n: "op" in gna(n) and gna(n)["op"] == "mc_store_op",
                    self.predecessors(mc),
                )
            )
            for mem_node in store_nodes:
                wire_state = f"{mem_node}.ndw0.state"
                if len(store_nodes) == 1:
                    return_buffer += f"INVAR (({wire_state}) = running);\n"

                store_request.append(f"({mem_node}.j.valid0)")
                store_grant.append(f"({wire_state} = running)")

            for id_ in range(len(store_request)):
                is_only_request = " & ".join(
                    [f"(!{m})" for m in store_request[:id_] + store_request[id_ + 1 :]]
                    + [store_request[id_]]
                )
                constr = f"({is_only_request} -> {store_grant[id_]})"
                return_buffer += f"INVAR {constr};\n"
                # TODO: when there are multiple store requests, one of
                # them must be granted for access. when there
                # are multiple store units, we say we
                # arbitrate them in random ways.

        for n, n_attr in filter(
            lambda n: n[1]["type"] == "Fork", self.nodes(data=True)
        ):
            out_edges = [e for e in filter(lambda e: e[0] == n, self.edges(data=True))]
            # look for fork -> ndw -> sink
            fns = []
            for e in filter(lambda e: gna(e[1])["type"] == "ndw", out_edges):
                if any(
                    [gna(v)["type"] == "Sink" for u, v in self.edges() if u == e[1]]
                ):
                    fns.append(e)
            for id_, (_, v, _) in enumerate(fns):
                if id_ > 0:
                    return_buffer += f"INVAR (({v}.state) = running);\n"
        return return_buffer
