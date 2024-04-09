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


## Tutorial: adding your own formal properties

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
