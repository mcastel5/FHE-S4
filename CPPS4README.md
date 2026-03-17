# OpenFHE Implementation of FHE-S4 Inference

This project contains a C++ implementation of FHE-S4 inference using the OpenFHE library. It computes the FHE-S4 sequence model by breaking the inference process into three phases, mixing Homomorphic Encryption (HE) and Plaintext computation.

## Overview

The inference process evaluates a subset of the components securely using the CKKS scheme in OpenFHE, while delegating non-linear activation layers (which are prohibitively expensive in HE) to a plaintext execution environment (e.g., Python/PyTorch).

The computation is broken down into three phases:

1. **Phase 1: FHE Toeplitz Convolution + Skip Connection**
   - Implemented in `openfhe_toeplitz.cpp`.
   - Performs a Toeplitz matrix multiplication (convolution) homomorphically using sequence shifting (`EvalAtIndex`) and accumulation.
   - Adds the skip connection homomorphically.
2. **Phase 2: Plaintext Non-linear Activation**
   - The encrypted state is decrypted and passed to a plaintext environment (e.g., PyTorch).
   - Computes non-linear functions (like GLU and expansions).
   - Simulated in the provided test pipeline via the Python export script.
3. **Phase 3: FHE Mean Reduction and Decoder**
   - The activated plaintext states are re-encrypted.
   - Computes mean reduction over the sequence length homomorphically using `EvalSum`.
   - Applies the linear decoder weights and bias to compute the final output.

## Prerequisites

- **OpenFHE**: Installed globally (e.g., headers in `/usr/local/include/openfhe` and libraries in `/usr/local/lib`).
- **CMake**: Version 3.16 or higher.
- **Python**: 3.x with `torch` and `numpy` installed (for generating test data).

> **Note**: Make sure your `LD_LIBRARY_PATH` includes the OpenFHE library path.  
> `export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`

## Workflow

### 1. Generate Test Data

First, generate the test inputs, convolution coefficients, expected skip connection outputs, and activated outputs using the Python script.

```bash
python export_for_openfhe.py --seq_len 32 --d_model 4 --toeplitz_K 16 --out toeplitz_test_data.json
```

This simulates the PyTorch model (`MiniS4D`) and exports all the context required by the C++ OpenFHE application.

### 2. Build the C++ Application

Use CMake to configure and build the OpenFHE application.

```bash
# Create a build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DOpenFHE_DIR=/usr/local/lib/OpenFHE

# Compile the executable
make -j$(nproc)
```

### 3. Run the FHE Evaluation

After compiling, run the `openfhe_toeplitz` executable, passing the JSON test data file generated in Step 1.

```bash
# Assuming you are still in the build/ directory
./openfhe_toeplitz ../toeplitz_test_data.json
```

The output will display the execution times for context creation, and the max absolute differences/errors for the FHE calculations in Phase 1 and Phase 3 against the expected plaintext outcomes.
