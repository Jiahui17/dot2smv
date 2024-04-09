from itertools import combinations, product
from textwrap import indent, dedent
from src.utils import include_guard, get_op_type
from src.dfg import DFG
import os, re, argparse, pprint
import pygraphviz as pgv
import networkx as nx

_tab = "\t"
_newline = "\n"


class ComponentWriter(nx.MultiDiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write_components(self):
        header = """
			// parametrized.smv
			#pragma once
			#include "./elastic_components.smv"
		"""
        module_description = set()
        module_description.add(header)
        module_description.add(write_mem(self))
        for node in self:
            node_attr = self.nodes.data()[node]

            np = len([e for e in self.edges if e[1] == node])  # number of predecessors

            ns = len([e for e in self.edges if e[0] == node])  # number of successors

            if node_attr["type"].lower() == "operator":
                match = re.search(
                    r"(operator|decider)([0-9]+)c", get_op_type(node_attr)
                )
                if match != None:
                    op = match.group(1)
                    delay = int(match.group(2))
                    if op == "operator":
                        module_description.add(write_operator(np, ns, delay))
                    elif op == "decider":
                        module_description.add(write_decider(np, ns, delay))

            elif node_attr["type"].lower() == "buffer":
                module_description.add(
                    write_buffer(node_attr["transparent"], int(node_attr["slots"]))
                )

            elif node_attr["type"].lower() == "merge":
                assert np in (
                    1,
                    2,
                )

            elif node_attr["type"].lower() == "fork" and ns != 2:
                module_description.add(write_fork(ns))

            elif node_attr["type"].lower() == "lazyfork":
                module_description.add(write_lazyfork(ns))

            elif node_attr["type"].lower() == "exit":
                module_description.add(write_exit(self))

        return "\n".join(list(module_description))


def write_mem(G):
    def gna(n):
        return G.nodes.data()[n]

    ret_buffer = ""
    for mem, attr in filter(
        lambda n: n[1]["type"] in ("LSQ", "MC"), G.nodes(data=True)
    ):
        ldcount = int(attr.get("ldcount", 0))
        stcount = int(attr.get("stcount", 0))
        bbcount = int(attr.get("bbcount", 0))
        memory = attr["memory"]
        type_ = attr["type"].lower()

        # get the input ports that are group requests
        l_group_req = [
            re.sub(r":.*", "", n)
            for n in attr.get("in", "").split()
            if re.search("c", n)
        ]

        # get the input ports that are address ports
        l_address_in = [
            re.sub(r":.*", "", n)
            for n in attr.get("in", "").split()
            if re.search("a", n)
        ]

        # get the input ports that are data ports
        l_data_in = [
            re.sub(":.*", "", n)
            for n in attr.get("in", "").split()
            if re.search("d", n)
        ]

        # get the output ports that are data ports
        l_data_out = [
            re.sub(":.*$", "", n)
            for n in attr.get("out", "").split()
            if re.search("d", n)
        ]

        # get the output ports that are address ports
        l_address_out = [
            re.sub(":.*$", "", n)
            for n in attr.get("out", "").split()
            if re.search("a", n)
        ]

        # get the output ports that are return ports
        l_end_out = [
            re.sub(":.*$", "", n)
            for n in attr.get("out", "").split()
            if re.search("e", n)
        ]
        # preamble
        n_in_entries = len(l_group_req + l_address_in + l_data_in)
        n_out_entries = len(l_data_out + l_address_out + l_end_out)

        ret_buffer += "_".join(
            (f"MODULE {type_}", memory, str(n_in_entries), str(n_out_entries))
        )

        ret_buffer += (
            "("
            + ", ".join(
                [f"dataIn{n}, pValid{n}" for n in range(n_in_entries)]
                + [f"nReady{n}" for n in range(n_out_entries)]
            )
            + ")\n"
        )

        for resp in l_data_out + l_address_out + l_end_out:
            resp_port = int(re.sub("out", "", resp)) - 1
            ret_buffer += "DEFINE dataOut" + str(resp_port) + " := FALSE;\n"
            ret_buffer += "DEFINE valid" + str(resp_port) + " := TRUE;\n"

        for group_req in l_group_req:
            if gna(mem)["type"] == "LSQ":
                req_port = int(re.sub("in", "", group_req)) - 1
                ret_buffer += f"""
				VAR gr_ndw_{req_port} : ndw_1_1 (FALSE, pValid{req_port}, TRUE);
				DEFINE ready{req_port} := gr_ndw_{req_port}.ready0;
			"""
            elif gna(mem)["type"] == "MC":
                req_port = int(re.sub("in", "", group_req)) - 1
                ret_buffer += f"""
				DEFINE ready{req_port} := TRUE;
			"""

        for address_in in l_address_in + l_data_in:
            req_port = int(re.sub("in", "", address_in)) - 1
            ret_buffer += f"""
			DEFINE ready{req_port} := FALSE;
		"""
    return ret_buffer


@include_guard
def write_fork(n_succ):

    n_succ = int(n_succ)
    if n_succ == 1:
        return """
		MODULE fork_1_1 (dataIn0, pValid0, nReady0)
			DEFINE dataOut0 := dataIn0;
			DEFINE valid0 := pValid0;
			DEFINE ready0 := nReady0;
			DEFINE ra0 := FALSE; // never run-ahead
			DEFINE efr0 := (TRUE); // EagerFork Register (remember)
			DEFINE efr_true0 := (dataIn0 -> efr0); // EagerFork Register (remember)
			DEFINE efr_false0 := ((!dataIn0) -> efr1); // EagerFork Register (remember)
		"""

    ret_buffer = f'MODULE fork_1_{n_succ}(dataIn0, pValid0, {", ".join(["nReady" + str(n) for n in range(n_succ)])})'
    ret_buffer += f'DEFINE forkStop := {" | ".join([f"regBlock{n}.blockStop" for n in range(n_succ)])};'
    ret_buffer += f"pValidAndForkStop := pValid0 & forkStop;\n"
    ret_buffer += f"ready0 := !forkStop;\n"
    for i in range(n_succ):
        ret_buffer += f"""
			VAR regBlock{i} : eagerFork_RegisterBlock(pValid0, nStop{i}, pValidAndForkStop);
			DEFINE nStop{i}      := !nReady{i};
			DEFINE valid{i}      := regBlock{i}.valid;
			DEFINE dataOut{i}    := dataIn0;
			DEFINE sent{i}       := !regBlock{i}.reg_value; // output i is running-ahead of the other blocks
			DEFINE sent_plus{i}  := !regBlock{i}.reg_value & dataIn0; 
			DEFINE sent_minus{i} := !regBlock{i}.reg_value & !dataIn0; 
			DEFINE efr{i} := (regBlock{i}.reg_value); // output i is not running-ahead of the other blocks
			DEFINE efr_plus{i} := (dataIn0 -> regBlock{i}.reg_value); // 
			DEFINE efr_minus{i} := ((!dataIn0) -> regBlock{i}.reg_value); // 
		"""
    return ret_buffer


@include_guard
def write_lazyfork(n_succ):
    n_succ = int(n_succ)
    if n_succ == 1:
        return """
		MODULE lazyfork_1_1 (dataIn0, pValid0, nReady0)
			DEFINE dataOut0 := dataIn0;
			DEFINE valid0 := pValid0;
			DEFINE ready0 := nReady0;
		"""
    ret_buffer = f'MODULE lazyfork_1_{n_succ}(dataIn0, pValid0, {", ".join(["nReady" + str(n) for n in range(n_succ)])})'
    ret_buffer += (
        f"DEFINE allnready := "
        + " & ".join([f"nReady{d}" for d in range(n_succ)])
        + ";\n"
    )
    ret_buffer += f"DEFINE ready0 := allnready;\n"
    for i in range(n_succ):
        ret_buffer += f"DEFINE valid{i} := pValid0 & allnready;\n"
        ret_buffer += f"DEFINE dataOut{i} := dataIn0;\n"
    return ret_buffer


@include_guard
def write_merge(n_pred):
    n_pred = int(n_pred)
    return f"""
			// merge_{n_pred}_1.smv
			
			MODULE merge_{n_pred}_1({', '.join([f'dataIn{i}, pValid{i}' for i in range(n_pred)])}, nReady0)
			
			DEFINE
			// tehb for breaking the combinational path
			VAR b0 : tehb_1_1(tehb_data_in, tehb_pvalid, nReady0);
			DEFINE num := toint(b0.full);
			DEFINE dataOut0 := b0.dataOut0;
			DEFINE valid0   := b0.valid0;
			
			// ready-i -> predecessors will be asserted when tehb is ready
			{''.join([f'{_newline}{_tab}ready{i} := b0.ready0;{_newline}' for i in range(n_pred)])}
			// valid -> successor will be asserted when either one of the input is ready
			
			tehb_pvalid := {' | '.join([f'pValid{i}' for i in range(n_pred)])};
			
			// and we priority muxing the output
			tehb_data_in := case
			{''.join([f'{_newline + _tab + _tab}pValid{i} : dataIn{i};{_newline}' for i in range(n_pred)])}
				TRUE : dataIn0;
			esac;
		"""


# exit node: takes M return-nodes and N memory-nodes
def write_exit(G):

    # find the operator nodes
    lop = [
        node for node in G.nodes if (G.nodes.data()[node]["type"]).lower() == "operator"
    ]

    # find list of return nodes (ret_op)
    lret = [node for node in lop if G.nodes.data()[node]["op"] == "ret_op"]

    # find the memory nodes (MC or LSQ)
    lmem = [
        node
        for node in G.nodes
        if (G.nodes.data()[node]["type"]).lower() in ("mc", "lsq")
    ]

    # number of predecessor nodes
    n_pred = len(lmem) + len(lret)

    exit_node = [n for n in G if G.nodes[n]["type"] == "Exit"][0]

    l_succ = [n for n in G.successors(exit_node)]
    n_succ = len(l_succ)

    # TODO: the ordering of in ports should be consistent with the information in the edge attributes
    # (not the order of how the nodes are loaded from the .dot file into the MultiDiGraph)
    ret_buffer = ""
    ret_buffer += f"""
		MODULE exit_{n_pred}_{n_succ}({', '.join( ['dataIn' + str(i) + ', pValid' + str(i) for i in range(n_pred)] + ['nReady' + str(i) for i in range(n_succ)] )})
	"""
    # whenever value returned from a return node, increase the inner token by 1
    ret_buffer += f"""
		DEFINE program_return := case
	"""
    for i in range(len(lmem), n_pred):
        ret_buffer += f"""
		pValid{i} : TRUE;
	"""
    ret_buffer += """
		TRUE : FALSE;
	esac;
	"""
    if n_succ == 0:
        ret_buffer += f"""
			VAR b0 : tehb_1_1(TRUE, program_return, FALSE);
			DEFINE num := b0.num;
			DEFINE full := b0.full;
		"""
    elif n_succ == 1:
        ret_buffer += f"""
			VAR b0 : tehb_1_1(TRUE, program_return, nReady0);
			DEFINE valid0 := b0.valid0;
			DEFINE dataOut0 := TRUE;
			DEFINE num := b0.num;
			DEFINE full := b0.full;
		"""
    else:
        assert False
    for i in range(len(lmem)):
        ret_buffer += f"""
			DEFINE ready{i} := FALSE;
		"""
    for i in range(len(lmem), n_pred):
        ret_buffer += f"""
			DEFINE ready{i} := b0.ready0;
		"""

    return ret_buffer


@include_guard
def write_fifo_inner(slots):
    # preamble:
    ret_buffer = f"""
		MODULE elasticFifoInner{slots}_1_1(dataIn0, pValid0, nReady0)
	"""
    # buffer slots:
    for i in range(slots):
        ret_buffer += f"""
		VAR mem_{i} : boolean; // buffer slots
	"""
    ret_buffer += f"""
		VAR wr_ptr : 0..({slots} - 1); // write pointer
		
		VAR rd_ptr : 0..({slots} - 1); // read pointer
		
		VAR empty   : boolean; // flags
		VAR full    : boolean;
		
		DEFINE rden     := nReady0 & valid0;
		DEFINE wren     := pValid0 & ready0;
		
		DEFINE valid0   := !empty;
		DEFINE ready0   := !full | nReady0; 
	"""
    ret_buffer += f"""
		DEFINE dataOut0 := case // output data:
	"""
    for i in range(slots):
        ret_buffer += f"""
		rd_ptr = {i} : mem_{i};
	"""
    else:
        ret_buffer += f"""
		TRUE : mem_0;
		esac;
	"""
    ret_buffer += f"""
		ASSIGN // write ptr update
		init(wr_ptr) := 0; 
		next(wr_ptr) := case
	"""
    for i in range(slots):
        ret_buffer += f"""
		wren & (wr_ptr = {i}) : {(i + 1) % slots};
	"""
    else:
        ret_buffer += f"""
		!wren : wr_ptr;
		TRUE : 0;
		esac;
	"""
    ret_buffer += f"""
		ASSIGN // read ptr update
		init(rd_ptr) := 0; 
		next(rd_ptr) := case
	"""
    for i in range(slots):
        ret_buffer += f"""
		rden & (rd_ptr = {i}) : {(i + 1) % slots};
	"""
    else:
        ret_buffer += f"""
		!rden : rd_ptr;
		TRUE : 0;
		esac;
	"""
    ret_buffer += f"""
		ASSIGN // full ptr update
		init(full)   := FALSE;
		next(full)   := case
		(wren & !rden & (wr_ptr + 1) mod {slots} = rd_ptr) : TRUE; // write, not read, and next position of write will overlap with read;
		(!wren & rden) : FALSE; // not write but read from the fifo:
		TRUE : full; // everything else doesn't change the full flag
		esac;
		
		ASSIGN // empty ptr update
		init(empty)  := TRUE;
		next(empty)  := case
		(!wren & rden & (rd_ptr + 1) mod {slots} = wr_ptr) : TRUE; // not write, read, and next position of read will overlap with write;
		(wren & !rden) : FALSE; // write but not read to the fifo:
		TRUE : empty; // everything else doesn't change the empty flag
		esac;
	"""
    for i in range(slots):
        ret_buffer += f"""
		ASSIGN init(mem_{i}) := FALSE;
		ASSIGN next(mem_{i}) := wren & (wr_ptr = {i}) ? dataIn0 : mem_{i};
	"""
    ret_buffer += f"""
		VAR num : 0..({slots}); -- fill level of the buffer
		ASSIGN init(num) := 0;
		ASSIGN next(num) := case
	"""
    for i in range(slots + 1):
        if i < slots:
            ret_buffer += f"""
				!rden & wren & (num = {i}) : {(i + 1) % (slots + 1)};
			"""
        if i > 0:
            ret_buffer += f"""
				rden & !wren & (num = {i}) : {(i - 1) % (slots + 1)};
			"""
    else:
        ret_buffer += f"""
		TRUE : num;
		esac;
	"""
    for i in range(slots):
        ret_buffer += f"""
		DEFINE used_{i} := ((rd_ptr < wr_ptr) & (rd_ptr <= {i}) & ({i} < wr_ptr) | ((rd_ptr > wr_ptr) & (rd_ptr <= {i} | {i} < wr_ptr)) | (full));
	"""

    # numplus_exists: flag that is set whenever any slot in the buffer has a valid TRUE
    numplus_exists = " | ".join([f"(used_{i} & mem_{i})" for i in range(slots)])
    ret_buffer += f"DEFINE numplus_exists := {numplus_exists};\n"

    # numminus_exists: flag that is set whenever any slot in the buffer has a valid FALSE
    numminus_exists = " | ".join([f"(used_{i} & !mem_{i})" for i in range(slots)])
    ret_buffer += f"DEFINE numminus_exists := {numminus_exists};\n"

    # ret_buffer += f'''
    # 	DEFINE numplus := count({", ".join([ f"used_{i} & mem_{i}" for i in range(slots) ])});
    # 	DEFINE numminus := count({", ".join([ f"used_{i} & !mem_{i}" for i in range(slots) ])});
    #'''

    # Note: the definition of numplus and numminus are used to track the occupancy of condition tokens in the buffers.
    # numplus: used to track how many valid tokens in the queue has a TRUE value
    ret_buffer += f"""
		VAR numplus : 0..({slots}); -- fill level of the buffer
		ASSIGN init(numplus) := 0;
		ASSIGN next(numplus) := case
	"""
    for i in range(slots + 1):
        if i < slots:
            ret_buffer += f"""
				!(rden & dataOut0) & (wren & dataIn0) & (numplus = {i}) : {(i + 1) % (slots + 1)};
			"""
        if i > 0:
            ret_buffer += f"""
				(rden & dataOut0) & !(wren & dataIn0) & (numplus = {i}) : {(i - 1) % (slots + 1)};
			"""
    else:
        ret_buffer += f"""
		TRUE : numplus;
		esac;
	"""
    # numminus: used to track how many valid tokens in the queue has a TRUE value
    ret_buffer += f"""
		VAR numminus : 0..({slots}); -- fill level of the buffer
		ASSIGN init(numminus) := 0;
		ASSIGN next(numminus) := case
	"""
    for i in range(slots + 1):
        if i < slots:
            ret_buffer += f"""
				!(rden & !dataOut0) & (wren & !dataIn0) & (numminus = {i}) : {(i + 1) % (slots + 1)};
			"""
        if i > 0:
            ret_buffer += f"""
				(rden & !dataOut0) & !(wren & !dataIn0) & (numminus = {i}) : {(i - 1) % (slots + 1)};
			"""
    else:
        ret_buffer += f"""
		TRUE : numminus;
		esac;
	"""
    return ret_buffer


def write_buffer(transparent, slots):
    if transparent == "true":
        transparent = True
    elif transparent == "false":
        transparent = False
    else:
        print(transparent)
        print(slots)
        assert False, "error - unknown input parameters for writing buffers"
    slots = int(slots)

    return "\n".join(
        (
            write_buffer_fifo_based(transparent, slots),
            write_buffer_slot_based(transparent, slots),
        )
    )


@include_guard
def write_buffer_fifo_based(transparent, slots):
    assert type(transparent) == bool and type(slots) == int and slots > 0
    if slots == 1 and not transparent:
        return f"""
			MODULE buffer1o_1_1(dataIn0, pValid0, nReady0)
			VAR
			b : elasticBuffer_1_1(dataIn0, pValid0, nReady0);
			DEFINE
			dataOut0 := b.dataOut0;
			valid0   := b.valid0;
			ready0   := b.ready0;
			num := count(b.b0.valid0, b.b1.full);
			numplus := count(b.b0.valid0 & b.b0.dataOut0, b.b1.full & b.b1.dataOut0);
			numminus := count(b.b0.valid0 & !b.b0.dataOut0, b.b1.full & !b.b1.dataOut0);
		"""
    elif slots == 1 and transparent:
        return f"""
			MODULE buffer1t_1_1(dataIn0, pValid0, nReady0)
			VAR
			b : tehb_1_1(dataIn0, pValid0, nReady0);
			DEFINE
			dataOut0 := b.dataOut0;
			valid0   := b.valid0;
			ready0   := b.ready0;
			num := count(b.full);
			numplus := count(b.full & b.dataOut0);
			numminus := count(b.full & !b.dataOut0);
		"""
    elif slots == 2 and not transparent:
        return f"""
			MODULE buffer2o_1_1(dataIn0, pValid0, nReady0)
			VAR
			b : elasticBuffer_1_1(dataIn0, pValid0, nReady0);
			DEFINE
			dataOut0 := b.dataOut0;
			valid0   := b.valid0;
			ready0   := b.ready0;
			num := count(b.b0.valid0, b.b1.full);
			numplus := count(b.b0.valid0 & b.b0.dataOut0, b.b1.full & b.b1.dataOut0);
			numminus := count(b.b0.valid0 & !b.b0.dataOut0, b.b1.full & !b.b1.dataOut0);
		"""
    elif slots >= 2 and transparent:
        ret_buffer = f"""
			MODULE buffer{slots}t_1_1(dataIn0, pValid0, nReady0)
			// if transparent, the connection would be 
			VAR fifoinner : elasticFifoInner{slots}_1_1(dataIn0, pValid0 & (!nReady0 | fifoinner.valid0), nReady0);
			DEFINE numplus := fifoinner.numplus;
			DEFINE numminus := fifoinner.numminus;
			DEFINE numplus_exists := fifoinner.numplus_exists;
			DEFINE numminus_exists := fifoinner.numminus_exists;
			
			DEFINE num := fifoinner.num;
			
			DEFINE valid0 := pValid0 | fifoinner.valid0;
			
			DEFINE ready0 := fifoinner.ready0 | nReady0;
			
			DEFINE dataOut0 := fifoinner.valid0 ? fifoinner.dataOut0 : dataIn0;
			{write_fifo_inner(slots)}
		"""
        return ret_buffer
    elif slots >= 3 and not transparent:
        ret_buffer = f"""
			MODULE buffer{slots}o_1_1(dataIn0, pValid0, nReady0)
			// if non-transparent, the connection would be tehb -> fifo
			DEFINE num := fifoinner.num + toint(tehb1.full);
			DEFINE numplus := fifoinner.numplus + toint(tehb1.full & tehb1.dataOut0);
			DEFINE numminus := fifoinner.numminus + toint(tehb1.full & !tehb1.dataOut0);
			DEFINE numplus_exists := fifoinner.numplus_exists;
			DEFINE numminus_exists := fifoinner.numminus_exists;
			VAR tehb1 : tehb_1_1(dataIn0, pValid0, fifoinner.ready0);
			
			VAR fifoinner : elasticFifoInner{slots}_1_1(tehb1.dataOut0, tehb1.valid0, nReady0);
			
			DEFINE dataOut0 := fifoinner.dataOut0;
			
			DEFINE valid0 := fifoinner.valid0;
			
			DEFINE ready0 := tehb1.ready0;
			{write_fifo_inner(slots)}
		"""
        return ret_buffer


@include_guard
def write_buffer_slot_based(transparent, slots):
    assert type(transparent) == bool and type(slots) == int and slots > 0
    # slot-based elastic buffers: prev -> b0 -> b1 -> b2 -> next
    if slots == 1 and not transparent:
        return f"""
			MODULE _buffer1o_1_1(dataIn0, pValid0, nReady0)
			VAR
			b0     : tehb_1_1(dataIn0, pValid0, b1.ready0);
			b1     : oehb_1_1(b0.dataOut0, b0.valid0, nReady0);
			DEFINE 
			dataOut0 := b1.dataOut0;
			valid0 := b1.valid0;
			ready0 := b0.ready0;
		"""
    elif slots == 1 and transparent:
        return f"""
			MODULE _buffer1t_1_1(dataIn0, pValid0, nReady0)
			VAR
			b0 : tehb_1_1(dataIn0, pValid0, nReady0);
			DEFINE
			dataOut0 := b0.dataOut0;
			valid0   := b0.valid0;
			ready0   := b0.ready0;
		"""
    elif slots == 2 and not transparent:
        return f"""
			MODULE _buffer2o_1_1(dataIn0, pValid0, nReady0)
			VAR
			b0     : tehb_1_1(dataIn0, pValid0, b1.ready0);
			b1     : oehb_1_1(b0.dataOut0, b0.valid0, nReady0);
			DEFINE 
			dataOut0 := b1.dataOut0;
			valid0 := b1.valid0;
			ready0 := b0.ready0;
		"""
    elif slots >= 2 and transparent:
        # in this case, the buffer will be implemented as cascading tslots
        ret_buffer = f"""MODULE _buffer{slots}t_1_1(dataIn0, pValid, nReady0)
		DEFINE dataOut0 := b{slots - 1}.dataOut0;
		DEFINE valid0   := b{slots - 1}.valid0;
		DEFINE ready0   := b0.ready0;
		"""
        slots_data = ["dataIn0"] + [f"b{i}.dataOut0" for i in range(slots - 1)]
        slots_valid = ["pValid"] + [f"b{i}.valid0" for i in range(slots - 1)]
        slots_ready = [f"b{i + 1}.ready0" for i in range(slots - 1)] + ["nReady0"]
        assert len(slots_data) == slots
        assert len(slots_valid) == slots
        assert len(slots_ready) == slots
        for id_, (data, valid, ready) in enumerate(
            zip(slots_data, slots_valid, slots_ready)
        ):
            ret_buffer += f"""
				VAR b{id_} : tslot_1_1({data}, {valid}, {ready});
			"""
        return ret_buffer
    elif slots >= 2 and not transparent:
        # NOTE: non transparent FIFO has 1 more slot than "slots"
        # in this case, the buffer will be implemented as 1 TEHB, slots - 1 tslots, and 1 OEHB
        ret_buffer = f"""MODULE _buffer{slots}o_1_1(dataIn0, pValid, nReady0)
		DEFINE dataOut0 := b{slots}.dataOut0;
		DEFINE valid0   := b{slots}.valid0;
		DEFINE ready0   := b0.ready0;
		"""
        slots_data = ["dataIn0"] + [f"b{i}.dataOut0" for i in range(slots)]
        slots_valid = ["pValid"] + [f"b{i}.valid0" for i in range(slots)]
        slots_ready = [f"b{i + 1}.ready0" for i in range(slots)] + ["nReady0"]
        slots_type = ["tehb"] + ["tslot" for _ in range(slots - 1)] + ["oehb"]
        assert len(slots_data) == slots + 1
        assert len(slots_valid) == slots + 1
        assert len(slots_ready) == slots + 1
        assert len(slots_type) == slots + 1
        for id_, (data, valid, ready, type_) in enumerate(
            zip(slots_data, slots_valid, slots_ready, slots_type)
        ):
            ret_buffer += f"""
				VAR b{id_} : {type_}_1_1({data}, {valid}, {ready});
		"""
        return ret_buffer
    else:
        assert False


def write_delay(latency):
    ret_buffer = f"""
		#ifndef __DELAY{latency}C_1_1
		#define __DELAY{latency}C_1_1
		MODULE delay{latency}c_1_1 (dataIn0, pValid0, nReady0)
	"""
    if latency == 0:
        ret_buffer += f"""
		DEFINE
		dataOut0 := dataIn0;
		valid0   := pValid0;
		ready0   := nReady0;
		num      := 0;
	"""
    elif latency == 1:
        ret_buffer += f"""
		VAR
		oehb0 : oehb_1_1(dataIn0, pValid0, nReady0);
		DEFINE
		dataOut0 := oehb0.dataOut0;
		valid0   := oehb0.valid0;
		ready0   := oehb0.ready0;
		num      := toint(oehb0.valid0);
		v1.full  := oehb0.valid0;
		v1.num   := toint(oehb0.valid0);
	"""
    elif latency >= 2:
        ret_buffer += f"""
		VAR oehb0 : oehb_1_1(dataIn0, v{latency - 1}, nReady0);
		DEFINE v0 := pValid0;
	"""
        for i in range(latency - 1):
            ret_buffer += f"""
		VAR v{i + 1} : boolean;
		ASSIGN init(v{i + 1}) := FALSE;
		ASSIGN next(v{i + 1}) := oehb0.ready0 ? v{i} : v{i + 1};
	"""
        ret_buffer += f"""
		DEFINE dataOut0 := FALSE;
		DEFINE valid0   := oehb0.valid0;
		DEFINE ready0   := oehb0.ready0;
		DEFINE v{latency}   := oehb0.valid0;
	"""
        slot_symbols = ", ".join(
            ["oehb0.valid0"] + [f"v{n}" for n in range(1, latency)]
        )
        ret_buffer += f"""
		DEFINE num := count({slot_symbols});
		"""
        for i in range(latency):
            ret_buffer += f"""
			DEFINE v{i + 1}.num := toint(v{i + 1});
			DEFINE v{i + 1}.full := v{i + 1};
			"""
    else:
        assert False, "error - illegal value of latency %s" % latency

    ret_buffer += f"""
		#endif
	"""
    return ret_buffer


""" write parametrized join """


@include_guard
def write_join(n_pred):
    n_pred = int(n_pred)
    return_buffer = ""
    return_buffer += (
        f"MODULE _join_{n_pred}_1("
        + ", ".join([f"pValid{i}" for i in range(n_pred)])
        + ", nReady0)\n"
    )
    for i in range(n_pred):
        return_buffer += (
            f"DEFINE ready{i} := "
            + " & ".join(["nReady0"] + [f"pValid{j}" for j in range(n_pred) if i != j])
            + ";\n"
        )
    return_buffer += (
        f"DEFINE valid0 := " + " & ".join([f"pValid{i}" for i in range(n_pred)]) + ";\n"
    )
    return return_buffer


""" write parametrized operators """


@include_guard
def write_operator(n_pred, n_succ, latency):
    if n_succ != 1:
        return ""
    # write the outer module
    input_symbols = ", ".join(
        [f"dataIn{i}, pValid{i}" for i in range(n_pred)]
        + [f"nReady{i}" for i in range(n_succ)]
    )
    ret_buffer = f"""
		MODULE operator{latency}c_{n_pred}_{n_succ}({input_symbols})
		VAR
		d0 : delay{latency}c_1_1(FALSE, j0.valid0, nReady0);
		j0 : _join_{n_pred}_1({", ".join(["pValid" + str(i) for i in range(n_pred)] + ["d0.ready0"])});
		DEFINE
		valid0   := d0.valid0;
		dataOut0 := d0.dataOut0;
		num      := d0.num;
	"""
    for i in range(n_pred):
        ret_buffer += f"""
			ready{i} := j0.ready{i};
		"""
    ret_buffer += write_join(n_pred)  # write the parametrized join
    ret_buffer += write_delay(latency)  # write the delay line

    return ret_buffer


""" write parametrized decider """


@include_guard
def write_decider(n_pred, n_succ, latency):
    # NOTE: delay stage of decider is not reading from its predecessor
    if not (n_pred == 2 and n_succ == 1):
        return ""

    # write the delay line
    ret_buffer = write_delay(latency)

    # write the outer module
    input_symbols = ", ".join(
        [f"dataIn{i}, pValid{i}" for i in range(n_pred)]
        + [f"nReady{i}" for i in range(n_succ)]
    )
    ret_buffer += f"""
		MODULE decider{latency}c_{n_pred}_{n_succ}({input_symbols})
		VAR
		j0   : join_2_1 (pValid0, pValid1, d0.ready0);
		d0   : delay{latency}c_1_1 (FALSE, j0.valid0, ndd0.ready0);
		ndd0 : ndd_1_1 (d0.dataOut0, d0.valid0, nReady0);

		DEFINE
		valid0   := ndd0.valid0;
		dataOut0 := ndd0.dataOut0;
		ready0   := j0.ready0;
		ready1   := j0.ready1;
		num      := d0.num; 

		MODULE redecider{latency}c_{n_pred}_{n_succ}({input_symbols})
		VAR
		j0   : join_2_1 (pValid0, pValid1, d0.ready0);
		d0   : delay{latency}c_1_1 (FALSE, j0.valid0, ndd0.ready0);
		ndd0 : rendd_1_1 (d0.dataOut0, d0.valid0, nReady0);

		DEFINE
		valid0   := ndd0.valid0;
		dataOut0 := ndd0.dataOut0;
		ready0   := j0.ready0;
		ready1   := j0.ready1;
		num      := d0.num; 
	"""
    return ret_buffer
