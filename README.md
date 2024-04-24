# A Model Generator Dataflow Circuit Verification

## Getting started (Linux)

Install the dependencies:

```sh
sudo apt-get update
sudo apt-get install git python3 graphviz graphviz-dev
pip install networkx pygraphviz
```

[Download
nuXmv](https://nuxmv.fbk.eu/theme/download.php?file=nuXmv-2.0.0-linux64.tar.gz).

Then unzip it to the directory of this README file: you should see something
like `dot2smv/nuXmv-2.0.0-Linux/bin/nuXmv`

To make sure everything works, please run the test examples:

```sh
cd test/fir
bash test.sh # takes about 5 seconds

cd test/iir
bash test.sh # takes about 10 seconds

cd test/matvec
bash test.sh # takes about 90 seconds
```

The test script `test/*/test.sh` generates the nuXmv model file and a script
file, and invoke nuXmv to perform model checking.

After running the script for each example, you could find a file `property.rpt`
in each directory, which lists the proven/disproven properties.


## Tutorial 1: adding your own formal properties

By default, the model generator `dot2smv` generates the absence of backpressure
properties. You might want to customize the set of generated properties.  Let's
see how to do this manually.


### 1. Disable default property generation

First we might want to disable the generation of default properties, we can
simply remove the generation statement in `dot2smv/dot2smv`:

```diff
--- prop_text = check_valid_not_ready(dfg)
+++ prop_text = "\n"
```

### 2. Generate the verification model using dot2smv

```sh
dot2smv test/fir/fir_optimized.dot
```

### 3. Manually adding your own properties

Then go to the directory `test/fir`, open the model file using any text editor,
go to the very end of the file, and add your property, for example:

```diff
INVARSPEC (!fork_0.valid0 | !fork_0.valid1);
```

Then, run the verification by invoking nuXmv:

```sh
../../nuXmv-2.0.0-Linux/bin/nuXmv -source prove.cmd
```

We should see something like this:
```
-- invariant (!fork_0.valid0 | !fork_0.valid1)  is false
elapse: 0.61 seconds, total: 0.61 seconds
```


## Tutorial 2: generate counterexample traces

Model checking verifies the correctness of the properties, if the property
fails, it can generate a counterexample.  In this Tutorial, we show how to
visualize it:


```sh
$ cd test/fir
$ ../../dot2smv fir_optimized.dot
$ ../../nuXmv-2.0.0-Linux/bin/nuXmv -source trace.cmd fir_optimized.dot
$ ls
fir_optimized.dot      model.smv     prove.cmd  trace.cmd
fir_optimized.dot.pdf  property.rpt  test.sh
$ ../../nuXmv-2.0.0-Linux/bin/nuXmv -source trace.cmd
```

With the commands above, nuXmv dumps the counterexamples in traces, in the
following files:

```sh
$ ls *_dbg_model.xml
1_dbg_model.xml  2_dbg_model.xml  3_dbg_model.xml
```

We can use the `traceparser` to convert the generated xml file to a PDF file,
which contains state-by-state visualization:

```sh
$ ../../traceparser fir_optimized.dot 1_dbg_model.xml # parse the traces
# you can view the counterexample trace by opening trace_1_dbg_model.pdf
```

To see which trace is for which property, we can checkout the file
`property.rpt`:

```
**** PROPERTY LIST [ Type, Status, Counter-example Number, Name ] ****
--------------------------  PROPERTY LIST  -------------------------
...
...
027 :(_Buffer_1.valid0 -> add_11.ready0) 
  [Invar          False          1      inv_no_stall__Buffer_1_nReadyArray_0]
028 :(_Buffer_2.valid0 -> fork_0.ready0) 
  [Invar          True           N/A    inv_no_stall__Buffer_2_nReadyArray_0]
029 :(_Buffer_3.valid0 -> branch_2.ready1) 
  [Invar          False          2      inv_no_stall__Buffer_3_nReadyArray_0]
030 :(_Buffer_4.valid0 -> branchC_5.ready0) 
  [Invar          True           N/A    inv_no_stall__Buffer_4_nReadyArray_0]
031 :(_Buffer_5.valid0 -> phi_1.ready0) 
  [Invar          False          3      inv_no_stall__Buffer_5_nReadyArray_0]
```

The labels 1, 2, or 3 indicate the trace labels. For instance, the property
with name `inv_no_stall__Buffer_1_nReadyArray_0` has a counterexample, stored
in trace `1_dbg_model.xml`.
