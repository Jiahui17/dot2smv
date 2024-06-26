'''
this files contains all non-parametrized smv models of dataflow units
'''


elastic_components = r'''
	// elastic_components.smv
	#pragma once

	MODULE sink_1_0(dataIn0, pValid0)
	DEFINE ready0 := TRUE;

	MODULE source_0_1(nReady0)
	DEFINE valid0  := TRUE;
	DEFINE dataOut0 := FALSE;

	////////////////////////////////////////
	// name     : open_0_1
	// inputs   : nReady0
	// outputs  : dataOut0, valid0
	///////////////////////////////////////

	MODULE open_0_1(nReady0)
	DEFINE
	valid0   := FALSE;
	dataOut0 := FALSE;

	////////////////////////////////////
	// name    : entry_0_1
	// inputs  : nReady0
	// outputs : dataOut0, valid0
	/////////////////////////////////////
	MODULE entry_0_1(nReady0)
	DEFINE dataOut0 := FALSE;
	VAR valid0   : boolean; // entry node resets with a (dummy) token
	ASSIGN init(valid0) := TRUE; // after firing it only has a bubble
	ASSIGN next(valid0) := (!nReady0 & valid0) ? TRUE : FALSE;
	DEFINE num := toint(valid0);
	DEFINE full := valid0;

	////////////////////////////////////
	// name    : entry_1_1
	// inputs  : nReady0
	// outputs : dataOut0, valid0
	/////////////////////////////////////

	MODULE entry_1_1(dataIn0, pValid0, nReady0)

	VAR 
	dataOut0 : boolean;
	valid0   : boolean;

	ASSIGN
	init(dataOut0) := FALSE;
	next(dataOut0) := enable ? dataIn0 : dataOut0;
	init(valid0)   := FALSE;
	next(valid0)   := pValid0 | (!ready0);

	DEFINE
	ready0 := (!valid0) | nReady0;
	enable := ready0 & pValid0;
	DEFINE num := toint(valid0);

	////////////////////////////////////
	// name    : receiver_1_0
	// inputs  : dataIn0, pValid0
	// outputs : 
	/////////////////////////////////////

	MODULE receiver_1_0(dataIn0, pValid0)
	// one-slot receiver
	VAR b0   : oehb_1_1(dataIn0, pValid0, FALSE);
	DEFINE ready0 := b0.ready0;
	DEFINE num := toint(b0.valid0);
	DEFINE full := b0.valid0;

	MODULE nd_sink_1_0(dataIn0, pValid0)
	VAR ndw0   : ndw_1_1(dataIn0, pValid0, TRUE);
	DEFINE ready0 := ndw0.ready0;
	MODULE channel_1_1(dataIn0, pValid0, nReady0)
	DEFINE
	dataOut0  := dataIn0;
	valid0    := pValid0;
	ready0    := nReady0;

	MODULE ndw_1_1 (dataIn0, pValid0, nReady0)
	VAR
	state : { sleeping, running };
	DEFINE
	dataOut0 := dataIn0;
	valid0   := state = running ? pValid0 : FALSE;
	ready0   := state = running ? nReady0 : FALSE;
	ASSIGN
	init(state) := { running, sleeping };
	next(state) := case
	state = sleeping & pValid0 : { sleeping, running };// when the wire is sleeping, it at least delays the data by one step; 
	state = running  & pValid0 : { sleeping, running };// when the wire is running, it at least passes through one valid data;
	TRUE : state;
	esac;
	FAIRNESS state=running;

	MODULE ndd_1_1(dataIn0, pValid0, nReady0)
	VAR
	random_bit : boolean;
	DEFINE
	dataOut0 := random_bit;
	valid0 := pValid0;
	ready0 := nReady0;
	ASSIGN
	init(random_bit) := {TRUE, FALSE};
	next(random_bit) := case
	valid0 & nReady0 : {TRUE, FALSE};
	TRUE : random_bit;
	esac;

	--| HACK: Below is a special implementation for the decider when it implements a loop bound condition
	--| This does not guarantee to work! If the first iterator value is not within the loop bound,
	--| it goes out immediately, and this model does not capture this.
	--| it implements a regular expression (1+0)*: e.g., 1111101110111110
	--| Here is how it works:
	--| - it resets with loop repeat condition (TRUE), i.e. they repeat the iteration at least once
	--| - if the current condition is loop repeat (TRUE), then the next condition is decided non-deterministically
	--| - if the current condition is loop exit (FALSE), then the next condition is a (loop repeat) TRUE

	MODULE rendd_1_1(dataIn0, pValid0, nReady0)
	VAR
	random_bit : boolean;
	DEFINE
	dataOut0 := random_bit;
	valid0 := pValid0;
	ready0 := nReady0;
	ASSIGN
	init(random_bit) := TRUE;
	next(random_bit) := case
	valid0 & nReady0 & (random_bit) : {TRUE, FALSE};
	valid0 & nReady0 & (!random_bit) : TRUE;
	TRUE : random_bit;
	esac;

	MODULE oehb_1_1(dataIn0, pValid0, nReady0)
	VAR 
	reg : boolean;
	valid0   : boolean;
	ASSIGN
	init(reg) := FALSE;
	next(reg) := enable ? dataIn0 : reg;
	init(valid0)   := FALSE;
	next(valid0)   := pValid0 | (!ready0);
	DEFINE
	dataOut0 := reg;
	ready0 := (!valid0) | nReady0;
	enable := ready0 & pValid0;
	num := toint(valid0);
	full := valid0;
	numplus  := toint(valid0 & reg);
	numminus := toint(valid0 & !reg);
	full_plus := full & dataOut0;
	full_minus := full & !dataOut0;

	MODULE tehb_1_1(dataIn0, pValid0, nReady0)
	VAR 
	reg     : boolean;
	full    : boolean;
	ASSIGN
	init(full)  := FALSE;
	next(full)  := valid0 & !nReady0;
	init(reg)  := FALSE;
	next(reg)  := enable ? dataIn0 : reg; 
	DEFINE
	valid0       := pValid0 | full; 
	ready0       := !full;
	enable      := ready0 & pValid0 & !nReady0;
	sel         := full;
	dataOut0     := sel ? reg : dataIn0;
	num := toint(full);
	numplus := count(full & dataOut0);
	numminus := count(full & !dataOut0);
	full_plus := full & dataOut0;
	full_minus := full & !dataOut0;

	MODULE elasticBuffer_1_1(dataIn0, pValid0, nReady0)
	VAR
	b0 : tehb_1_1(dataIn0, pValid0, b1.ready0);
	b1 : oehb_1_1(b0.dataOut0, b0.valid0, nReady0);
	DEFINE
	dataOut0 := b1.dataOut0;
	valid0   := b1.valid0;
	ready0   := b0.ready0;
	num := count(b1.valid0, b0.full);

	///////////////////////////////////////////
	// name     : tslot
	// inputs   : dataIn0, pValid0, nReady0
	// outputs  : dataOut0, valid0, ready0
	///////////////////////////////////////////

	MODULE tslot_1_1(dataIn0, pValid0, nReady0)
	// fully transparent slot
	VAR 
	reg      : boolean;
	full     : boolean;

	ASSIGN
	init(reg)    := FALSE;
	next(reg)    := enable ? dataIn0 : reg;
	init(full)   := FALSE;
	next(full)   := (full <-> nReady0) ? pValid0 : full;

	DEFINE
	valid0 := full | pValid0; // tslot has valid data whenever it is full, or pValid0
	ready0 := (!full) | nReady0; // tslot ready it is not full, or successor ready for receive data
	enable := pValid0 & (nReady0 <-> full); // load data whenever nReady0 & pValid0
	dataOut0 := full ? reg : dataIn0;
	num := toint(full);
	numplus := toint(full & reg);
	numminus := toint(full & !reg);
	full_plus := full & dataOut0;
	full_minus := full & !dataOut0;

	MODULE constant_1_1(dataIn0, pValid0, nReady0)
	DEFINE
	dataOut0 := dataIn0;
	valid0   := pValid0;
	ready0   := nReady0; 

	MODULE join_2_1 (pValid0, pValid1, nReady0)
	DEFINE
	valid0 := pValid0 & pValid1;
	ready0 := nReady0 & pValid1;
	ready1 := nReady0 & pValid0;

	MODULE join_3_1 (pValid0, pValid1, pValid2, nReady0)
	DEFINE ready0  := nReady0 & pValid1 & pValid2;
	DEFINE ready1  := nReady0 & pValid0 & pValid2;
	DEFINE ready2  := nReady0 & pValid0 & pValid1;
	DEFINE valid0  := pValid0 & pValid1 & pValid2;

	MODULE branchSimple(condition, pValid, nReady0, nReady1)
	DEFINE valid1 := !condition & pValid;
	DEFINE valid0 :=  condition & pValid;
	DEFINE ready  :=  nReady1 & !condition | nReady0 & condition;

	MODULE branch_2_2(dataIn0, pValid0, dataIn1, pValid1, nReady0, nReady1)
	/*
	port naming for branch
	input-0: data input      (dataIn0, pValid0, ready0)
	input-1: condition input (dataIn1, pValid1, ready1)
	*/
	DEFINE condition := dataIn1;
	VAR    j         : join_2_1(pValid0, pValid1, br.ready);
	VAR    br        : branchSimple(condition, j.valid0, nReady0, nReady1);
	DEFINE dataOut0  := dataIn0; // when ctrl = TRUE
	DEFINE dataOut1  := dataIn0; // when ctrl = FALSE
	DEFINE valid0    := br.valid0;
	DEFINE valid1    := br.valid1;
	DEFINE ready0 := j.ready0; // data input
	DEFINE ready1 := j.ready1; // ctrl input

	MODULE eagerFork_RegisterBlock(pValid, nStop, pValidAndForkStop)
	VAR
	reg_value : boolean;
	DEFINE
	blockStop := nStop & reg_value;
	valid := reg_value & pValid;
	ASSIGN
	init(reg_value) := TRUE;
	next(reg_value) := blockStop | !pValidAndForkStop;

	MODULE fork_1_2(dataIn0, pValid0, nReady0, nReady1)
	VAR regBlock0 : eagerFork_RegisterBlock(pValid0, nStop0, pValidAndForkStop);
	VAR regBlock1 : eagerFork_RegisterBlock(pValid0, nStop1, pValidAndForkStop);
	DEFINE forkStop := regBlock0.blockStop | regBlock1.blockStop;
	pValidAndForkStop := pValid0 & forkStop;
	DEFINE nStop0 := !nReady0;
	DEFINE nStop1 := !nReady1;
	ready0 := !forkStop;
	valid0 := regBlock0.valid;
	valid1 := regBlock1.valid;
	dataOut0 := dataIn0;
	dataOut1 := dataIn0;

	-- for constraint generation modules
	DEFINE sent0 := !regBlock0.reg_value; // f0 running-ahead
	DEFINE sent1 := !regBlock1.reg_value; // f1 running-ahead

	DEFINE sent_plus0   := !regBlock0.reg_value & dataIn0; 
	DEFINE sent_minus0  := !regBlock0.reg_value & !dataIn0; 
	DEFINE sent_plus1   := !regBlock1.reg_value & dataIn0;
	DEFINE sent_minus1  := !regBlock1.reg_value & !dataIn0;

	DEFINE efr0 := (regBlock0.reg_value); // f0 running-ahead
	DEFINE efr1 := (regBlock1.reg_value); // f1 running-ahead
	DEFINE efr_plus0 := (dataIn0 -> regBlock0.reg_value); 
	DEFINE efr_plus1 := (dataIn0 -> regBlock1.reg_value); 
	DEFINE efr_minus0 := ((!dataIn0) -> regBlock0.reg_value); 
	DEFINE efr_minus1 := ((!dataIn0) -> regBlock1.reg_value); 

	///////////////////////////////////////////////////////
	// module : merge_1_1
	// inputs : dataIn0, pValid0, nReady0
	// outputs: dataOut0, valid0, ready0
	//////////////////////////////////////////////////////

	MODULE merge_1_1 (dataIn0, pValid0, nReady0)
	DEFINE
	VAR b0 : tehb_1_1(tehb_data_in, pValid0, nReady0);
	DEFINE dataOut0 := b0.dataOut0;
	DEFINE valid0   := b0.valid0; 
	ready0 := b0.ready0;
	// 1-input merge with colored output: always FALSE (from BB that dominates it
	tehb_data_in := FALSE;
	DEFINE num := toint(b0.full); // convience

	///////////////////////////////////////////////////////
	// module : merge_2_1
	// inputs : dataIn0, pValid0, dataIn1, pValid1, nReady0
	// outputs: dataOut0, valid0, ready0, ready1
	//////////////////////////////////////////////////////

	MODULE merge_2_1(dataIn0, pValid0, dataIn1, pValid1, nReady0)
	DEFINE
	VAR b0 : tehb_1_1(tehb_data_in, tehb_pvalid, nReady0);
	DEFINE dataOut0 := b0.dataOut0;
	DEFINE valid0   := b0.valid0; 
	ready0 := b0.ready0;
	ready1 := b0.ready0; 
	tehb_pvalid := pValid0 | pValid1; 

	// new implementation: merge colors output based on which input it has received
	tehb_data_in := case 
	pValid0 : FALSE;
	pValid1 : TRUE;
	TRUE : FALSE;
	esac;

	// tehb_data_in := case 
	// pValid0 : dataIn0; 
	// pValid1 : dataIn1;
	// TRUE : dataIn0;
	// esac;

	DEFINE num := toint(b0.full); // convience


	MODULE merge_notehb_2_1(dataIn0, pValid0, dataIn1, pValid1, nReady0)
	DEFINE
	ready0 := nReady0;
	ready1 := nReady0;
	valid0 := pValid0 | pValid1;

	// new implementation: merge colors output based on which input it has received
	dataOut0 := case
	pValid0 : FALSE;
	pValid1 : TRUE;
	TRUE : FALSE;
	esac;

	// dataOut0 := case
	// pValid0 : dataIn0;
	// pValid1 : dataIn1;
	// TRUE : dataIn0;
	// esac;

	///////////////////////////////////////////////////////
	// module : cntrlmerge_2_2
	// inputs : dataIn0, pValid0, dataIn1, pValid1, nReady0, nReady1
	// outputs: dataOut0, valid0, dataOut1, valid1, ready0, ready1
	//////////////////////////////////////////////////////

	MODULE cntrlmerge_2_2(dataIn0, pValid0, dataIn1, pValid1, nReady0, nReady1)
	DEFINE dataOut0 := f0.dataOut0;
	DEFINE ready0 := m0.ready0;
	DEFINE dataOut1 := b0.dataOut0;
	DEFINE ready1 := m0.ready1;
	DEFINE index := case
	pValid0 : FALSE;
	TRUE    : TRUE;
	esac;
	VAR b0 : tehb_1_1(index, m0.valid0, f0.ready0); 
	VAR m0 : merge_notehb_2_1(dataIn0, pValid0, dataIn1, pValid1, b0.ready0);
	VAR f0 : fork_1_2(b0.dataOut0, b0.valid0, nReady0, nReady1);
	DEFINE valid0 := f0.valid0;
	DEFINE valid1 := f0.valid1;
	DEFINE efr0 := f0.efr0; // f0 running-ahead // convience
	DEFINE efr1 := f0.efr1; // f1 running-ahead // convience

	DEFINE efr_plus0  := (b0.dataOut0 -> efr0);
	DEFINE efr_plus1  := (b0.dataOut0 -> efr1);
	DEFINE efr_minus0 := (!b0.dataOut0 -> efr0);
	DEFINE efr_minus1 := (!b0.dataOut0 -> efr1);

	-- DEFINE efr_plus0 := f0.efr_plus0; // convience
	-- DEFINE efr_plus1 := f0.efr_plus1; // convience
	-- DEFINE efr_minus0 := f0.efr_minus0; // convience
	-- DEFINE efr_minus1 := f0.efr_minus1; // convience

	DEFINE mem := b0.reg; // convience
	DEFINE num := toint(b0.full); // convience
	DEFINE numplus := toint(b0.full & b0.dataOut0); // convience
	DEFINE numminus := toint(b0.full & !b0.dataOut0); // convience

	///////////////////////////////////////////////////////
	// module : mux_3_1
	// inputs : dataIn0, pValid0, dataIn1, pValid1, dataIn2, pValid2, nReady0
	// outputs: dataOut0, valid0, ready0, ready1, ready2
	//////////////////////////////////////////////////////

	MODULE mux_3_1(dataIn0, pValid0, dataIn1, pValid1, dataIn2, pValid2, nReady0)
	DEFINE sel := dataIn0;

	// new implementation: Mux colors output based on its selection input
	DEFINE tehb_data_in := dataIn0;

	// old implementation: Mux propagates data (which will be removed by coi)
	// DEFINE tehb_data_in := case
	// pValid0 & sel = FALSE & pValid1 : dataIn1; // if sel-valid, sel-data = 0,  left-pred-ready: take left input
	// pValid0 & sel = TRUE  & pValid2 : dataIn2; // if sel-valid, sel-data = 1, right-pred-ready: take right input
	// TRUE : dataIn1; // everything else, for instance sel-data is not ready
	// esac;

	DEFINE ready1 := (!sel & pValid0 & b0.ready0 & pValid1) | !pValid1 ? TRUE : FALSE;
	DEFINE ready2 := (sel & pValid0 & b0.ready0 & pValid2) | !pValid2 ? TRUE : FALSE;
	DEFINE tehb_pvalid := (pValid0 & !sel & pValid1) | (pValid0 & sel & pValid2) ? TRUE : FALSE;
	VAR b0 : tehb_1_1(tehb_data_in, tehb_pvalid, nReady0);
	DEFINE valid0 := b0.valid0;
	DEFINE dataOut0 := b0.dataOut0;
	DEFINE ready0 := (!pValid0 | tehb_pvalid & b0.ready0);
	DEFINE num := toint(b0.full); // convience

--	---------------------------------------------------------------------------
--	-- New description of select operator
--	---------------------------------------------------------------------------
--
--	MODULE select_op_3_1(dataIn0, pValid0, dataIn1, pValid1, dataIn2, pValid2, nReady0)
--
--	VAR
--
--	antitoken1 : boolean;
--	antitoken2 : boolean;
--
--	DEFINE
--
--	sel := dataIn0;
--
--	-- valid data to send 
--	ee := sel ? pValid0 & pValid1 : pValid0 & pValid2;
--
--	-- transfer is allowed only if the antitokens from the previous round are all used.
--	valid0 := ee & !antitoken1 & !antitoken2;
--
--	out_transfer := valid0 & nReady0;
--
--	-- ready to accept data when transfer is possible
--	ready0 := !pValid0 | out_transfer;
--	-- ready to accept data when transfer is possible, or antitoken can cancel discarded token
--	ready1 := !pValid1 | out_transfer | antitoken1;
--	-- ready to accept data when transfer is possible, or antitoken can cancel discarded token
--	ready2 := !pValid2 | out_transfer | antitoken2;
--
--	dataOut0 := sel ? dataIn1 : dataIn2;
--
--	ASSIGN
--	init(antitoken1) := FALSE;
--	init(antitoken2) := FALSE;
--
--	next(antitoken1) := !pValid1 & (antitoken1 | out_transfer);
--	next(antitoken2) := !pValid2 & (antitoken2 | out_transfer); 

	---------------------------------------------------------------------------
	-- Old description of select operator
	---------------------------------------------------------------------------

	MODULE select_op_3_1(dataIn0, pValid0, dataIn1, pValid1, dataIn2, pValid2, nReady0)
	VAR 
	anti0 : antitokens(pValid2, pValid1, generate_at1, generate_at0);
	DEFINE 
	sel := dataIn0;// selection signal : input-0 
	ee  := sel ? pValid0 & pValid1 : pValid0 & pValid2;// select observes valid data to transfer 
	valid0 := ee & !anti0.stop_valid;// select can propagate new token only after it killed the unused token from previous
	// generate commands: if the unused path is not killed at the same cycle where used path is accpted, then forward this info to the antitokens
	generate_at0 := !pValid1 & valid0 & nReady0; 
	generate_at1 := !pValid2 & valid0 & nReady0;
	ready0 := !pValid0 | (valid0 & nReady0);// like normal join
	ready1 := !pValid1 | (valid0 & nReady0) | anti0.kill0; // like normal join, or kill the unused input
	ready2 := !pValid2 | (valid0 & nReady0) | anti0.kill1; // like normal join, or kill the unused input 
	dataOut0 := sel ? dataIn1 : dataIn2;
	antitoken1 := anti0.reg_out0;
	antitoken2 := anti0.reg_out1;

	MODULE antitokens(pValid1, pValid0, generate_at1, generate_at0)
	VAR
	reg_out0 : boolean;
	reg_out1 : boolean;
	DEFINE
	kill0      := generate_at0 | reg_out0;
	kill1      := generate_at1 | reg_out1;
	stop_valid :=     reg_out0 | reg_out1;
	ASSIGN
	init(reg_out0) := FALSE;
	init(reg_out1) := FALSE;
	next(reg_out0) := !pValid0 & (generate_at0 | reg_out0);
	next(reg_out1) := !pValid1 & (generate_at1 | reg_out1);

	-----------------------------
	-- select unit with counter
	-----------------------------
	MODULE select_op_counter_3_1(dataIn0, pValid0, dataIn1, pValid1, dataIn2, pValid2, nReady0)
	VAR 
	counter1 : 0..31;
	anti0 : antitokens(pValid2, pValid0_temp, generate_at1, generate_at0);

	DEFINE 
	ee  := pValid0 & ((send_internal_1) | (sel & pValid1 & (counter1 = 0)));
	sel := dataIn0;
	valid0 := ee & !anti0.stop_valid;
	generate_at0 := !pValid1 & valid0 & nReady0; 
	generate_at1 := !pValid2 & valid0 & nReady0;
	ready1 := !pValid1 | (valid0 & nReady0) | kill0_temp; // like normal join, or kill the unused input
	ready2 := !pValid2 | (valid0 & nReady0) | anti0.kill1; // like normal join, or kill the unused input 
	ready0 := !pValid0 | (valid0 & nReady0);// like normal join
	dataOut0 := sel ? dataIn1 : dataIn2;

	// at least 1 token at '+' input to kill
	kill0_temp   := !(counter1 = 0);
	pValid0_temp := (counter1 < 31);
	counter_zero := (counter1 = 0);
	send_internal_0 := (kill0_temp & pValid1 & !send_internal_1);

	// propagating the '-' input to output
	// whenever
	// - required data are valid (pValid0 and pValid2)
	// - input2 is selected
	// - counter still has capacity (!)
	send_internal_1 := (pValid0 & !sel & pValid2 & (pValid0_temp));
	kill0 := anti0.kill0;
	antitoken1 := anti0.reg_out0;
	antitoken2 := anti0.reg_out1;

	ASSIGN
	init(counter1) := 0;
	next(counter1) := case
		kill0 & !send_internal_0 & pValid0_temp : counter1 + 1;
		!kill0 & send_internal_0 : counter1 - 1;
		TRUE : counter1;
	esac;


	-- this is used to model a shift register (used for instance, inside a multiplier)
	MODULE shift1c(dataIn0, pValid0, nReady0)
	VAR
		dataOut0 : boolean;
		valid0 : boolean;

	DEFINE
		ready0 := nReady0;
		full := valid0;
	
	ASSIGN
		init(valid0) := FALSE;
		next(valid0) := nReady0 ? pValid0 : valid0;
		init(dataOut0) := FALSE;
		next(dataOut0) := (pValid0 & nReady0) ? dataIn0 : dataOut0;

	MODULE mc_load_op_2_2(dataIn0, pValid0, dataIn1, pValid1, nReady0, nReady1)
	/*
	port naming:
	input-0  : (don't care) data read from MC node
	input-1  : address read from predecessor node
	output-0 : data output to the successor node
	output-1 : (don't care) address output to the MC node
	*/
	VAR b0 : tehb_1_1 (dataIn1, pValid1, ndw0.ready0);
	VAR ndw0  : ndw_1_1  (b0.dataOut0, b0.valid0, b1.ready0); 
	VAR b1 : shift1c (ndw0.dataOut0, ndw0.valid0, b2.ready0);
	VAR b2 : tehb_1_1 (b1.dataOut0, b1.valid0, nReady0);
	DEFINE dataOut0 := b2.dataOut0; // dataOut0 -> successor load output
	DEFINE valid0   := b2.valid0;   // valid0   -> successor load output
	DEFINE ready1   := b0.ready0;   // ready1   -> predecessor request
	DEFINE ready0   := FALSE;           // ready0   -> MC (don't care)
	DEFINE dataOut1 := FALSE;             // dataOut1 -> MC (don't care)
	DEFINE valid1   := TRUE;          // valid1   -> MC (don't care)
	DEFINE num      := count(b0.full, b1.valid0, b2.full);

	MODULE mc_store_op_2_2(dataIn0, pValid0, dataIn1, pValid1, nReady0, nReady1)
	/*
	port naming:
	input-0  : data input from store request predecessor
	input-1  : address input from store request predecessor
	output-0 : (don't care) data output to MC
	output-1 : (don't care) address output to MC
	*/
	VAR j     : join_2_1 (pValid0, pValid1, ndw0.ready0); 
	VAR ndw0  : ndw_1_1 (FALSE, j.valid0, sink0.ready0);
	VAR sink0 : sink_1_0 (ndw0.dataOut0, ndw0.valid0);
	DEFINE dataOut0 := FALSE;       // dataOut0 -> data output to MC (don't care)
	DEFINE valid0   := TRUE;    // valid0   -> data output to MC (don't care)
	DEFINE ready0   := j.ready0; // ready0   -> data input from store request predecessor
	DEFINE dataOut1 := FALSE;       // dataOut1 -> address output to MC successor (don't care)
	DEFINE valid1   := TRUE;    // valid1   -> address output to MC successor (don't care)
	DEFINE ready1   := j.ready1; // ready1   -> address input from store request predecessor
	DEFINE num      := 0;

	MODULE lsq_store_op_2_2(dataIn0, pValid0, dataIn1, pValid1, nReady0, nReady1)
	/* port naming:
	input-0  : data input from store request predecessor
	input-1  : address input from store request predecessor
	output-0 : (don't care) data output to LSQ
	output-1 : (don't care) address output to LSQ */
	VAR ndw0 : ndw_1_1 (dataIn0, pValid0, TRUE);// input-0 -> ndw0 -> sink0
	VAR ndw1 : ndw_1_1 (dataIn1, pValid1, TRUE);// input-1 -> ndw1 -> sink1
	DEFINE ready0   := ndw0.ready0; // ready0 -> data input from store request predecessor
	DEFINE ready1   := ndw1.ready0; // ready1 -> address input from store request predecessor
	DEFINE dataOut0 := FALSE;       // dataOut0 -> data output to LSQ (don't care)
	DEFINE valid0   := TRUE;     // valid0 -> data output to LSQ (don't care)
	DEFINE dataOut1 := FALSE;    // dataOut1 -> address output to LSQ successor (don't care)
	DEFINE valid1   := TRUE;     // valid1 -> address output to LSQ successor (don't care)

	MODULE lsq_load_op_2_2(dataIn0, pValid0, dataIn1, pValid1, nReady0, nReady1)
	// input-0   : (don't care) data read from LSQ node 
	// input-1   : address read from predecessor node
	// output-0  : data output to the successor node
	// output-1  : (don't care) address output to the LSQ node

	VAR
	ndw0  : ndw_1_1(dataIn1, pValid1, b0.ready0); 
	b0    : tslot_1_1(ndw0.dataOut0, ndw0.valid0, b1.ready0);
	b1    : tslot_1_1(b0.dataOut0,   b0.valid0,   b2.ready0);
	b2    : tslot_1_1(b1.dataOut0,   b1.valid0,   b3.ready0);
	b3    : tslot_1_1(b2.dataOut0,   b2.valid0,   b4.ready0);
	b4    : tslot_1_1(b3.dataOut0,   b3.valid0,   b5.ready0);
	b5    : oehb_1_1(b4.dataOut0,   b4.valid0,   ndw1.ready0);
	ndw1  : ndw_1_1(b5.dataOut0, b5.valid0, nReady0);

	DEFINE
	dataOut0 := ndw1.dataOut0; // dataOut0 -> successor load output
	valid0   := ndw1.valid0;   // valid0   -> successor load output
	ready1   := ndw0.ready0;   // ready1   -> predecessor request
	ready0   := FALSE;         // ready0   -> MC (don't care)
	dataOut1 := FALSE;         // dataOut1 -> MC (don't care)
	valid1   := TRUE;          // valid1   -> MC (don't care)
'''


logical_operators = '''

	MODULE and_op_2_1(dataIn0, pValid0, dataIn1, pValid1, nReady0)
	VAR
	j0 : join_2_1(pValid0, pValid1, nReady0);
	DEFINE
	dataOut0 := dataIn0 & dataIn1;
	ready0   := j0.ready0;
	ready1   := j0.ready1;
	valid0   := j0.valid0;

	MODULE or_op_2_1(dataIn0, pValid0, dataIn1, pValid1, nReady0)
	VAR
	j0 : join_2_1(pValid0, pValid1, nReady0);
	DEFINE
	dataOut0 := dataIn0 | dataIn1;
	ready0   := j0.ready0;
	ready1   := j0.ready1;
	valid0   := j0.valid0;

'''

compositional_components = '''
	MODULE _buffer2t_wt_1_1(dataIn0, pValid0, nReady0)
	VAR fifoinner : elasticCounterInner2_wt_1_1(dataIn0, pValid0 & (!nReady0 | fifoinner.valid0), nReady0);
	DEFINE valid0 := pValid0 | fifoinner.valid0;
	DEFINE ready0 := fifoinner.ready0 | nReady0;
	DEFINE dataOut0 := fifoinner.valid0 ? fifoinner.dataOut0 : dataIn0;

	MODULE elasticCounterInner2_wt_1_1(dataIn0, pValid0, nReady0)
	VAR counter : 0..(2);
	DEFINE empty := (counter = 0);
	DEFINE full  := (counter = (2));
	DEFINE valid0 := !empty;
	DEFINE ready0 := (full -> nReady0);
	DEFINE dataOut0 := FALSE;
	DEFINE rden     := nReady0 & valid0;
	DEFINE wren     := pValid0 & ready0;
	ASSIGN init(counter) := 1;
	ASSIGN next(counter) := case
	!rden & wren & (counter = 0) : 1;
	!rden & wren & (counter = 1) : 2;
	rden & !wren & (counter = 1) : 0;
	rden & !wren & (counter = 2) : 1;
	TRUE : counter;
	esac;

	MODULE _buffer2o_wt_1_1(dataIn0, pValid0, nReady0)
	VAR
	b : elasticBuffer_wt_1_1(dataIn0, pValid0, nReady0);
	DEFINE
	dataOut0 := b.dataOut0;
	valid0   := b.valid0;
	ready0   := b.ready0;

	MODULE elasticBuffer_wt_1_1(dataIn0, pValid0, nReady0)
	VAR
	tail_buffer : tehb_1_1(dataIn0, pValid0, head_buffer.ready0);
	head_buffer : oehb_WT_1_1(tail_buffer.dataOut0, tail_buffer.valid0, nReady0);
	DEFINE
	dataOut0 := head_buffer.dataOut0;
	valid0   := head_buffer.valid0;
	ready0   := tail_buffer.ready0;

	MODULE oehb_WT_1_1(dataIn0, pValid0, nReady0)
	VAR
	dataOut0 : boolean;
	valid0   : boolean;

	ASSIGN
	init(dataOut0) := FALSE;
	next(dataOut0) := enable ? dataIn0 : dataOut0;
	init(valid0)   := TRUE;
	next(valid0)   := pValid0 | (!ready0);
	DEFINE
	ready0  := (!valid0) | nReady0;
	enable  := ready0 & pValid0;

'''

obsolete_components = '''
	MODULE _lsq_load_op_2_2(dataIn0, pValid0, dataIn1, pValid1, nReady0, nReady1)
	/* port naming:
	input-0   : (don't care) data read from LSQ node 
	input-1   : address read from predecessor node
	output-0  : data output to the successor node
	output-1  : (don't care) address output to the LSQ node */
	VAR ndw0  : ndw_1_1(dataIn1, pValid1, fifo0.ready0); // block diagram of a simplified lsq_load_node // ndw0 -> fifo0 -> ndw1
	VAR fifo0 : _lsq_count(ndw0.dataOut0, ndw0.valid0, ndw1.ready0); // VAR fifo0 : fifo_lsq_load_inner_1_1(ndw0.dataOut0, ndw0.valid0, ndw1.ready0);
	VAR ndw1  : ndw_1_1(fifo0.dataOut0, fifo0.valid0, nReady0);
	DEFINE dataOut0 := ndw1.dataOut0; // dataOut0 -> successor load output
	DEFINE valid0   := ndw1.valid0;   // valid0   -> successor load output
	DEFINE ready1   := ndw0.ready0;   // ready1   -> predecessor request
	DEFINE ready0   := FALSE;          // ready0   -> MC (don't care)
	DEFINE dataOut1 := FALSE;             // dataOut1 -> MC (don't care)
	DEFINE valid1   := TRUE;          // valid1   -> MC (don't care)
	DEFINE num      := fifo0.num;

	#define queue_depth 6
	MODULE _lsq_count(dataIn0, pValid0, nReady0)
	VAR counter : 0..(queue_depth);
	DEFINE empty := (counter = 0);
	DEFINE full  := (counter = (queue_depth));
	DEFINE valid0 := !empty;
	DEFINE ready0 := (full -> nReady0);
	DEFINE dataOut0 := FALSE;
	DEFINE rden := nReady0 & valid0;
	DEFINE wren := pValid0 & ready0; 
	DEFINE num  := counter;
	ASSIGN init(counter) := 0;
	ASSIGN next(counter) := case
		!rden & wren & (counter = 0) : 1;
		!rden & wren & (counter = 1) : 2;
		!rden & wren & (counter = 2) : 3;
		!rden & wren & (counter = 3) : 4;
		!rden & wren & (counter = 4) : 5;
		!rden & wren & (counter = 5) : 6;
						    
		rden & !wren & (counter = 1) : 0;
		rden & !wren & (counter = 2) : 1;
		rden & !wren & (counter = 3) : 2;
		rden & !wren & (counter = 4) : 3;
		rden & !wren & (counter = 5) : 4;
		rden & !wren & (counter = 6) : 5;

		--!rden & wren : (counter + 1) mod queue_depth;
		--rden & !wren : (counter - 1) mod queue_depth;
		TRUE : counter;
	esac;
'''
