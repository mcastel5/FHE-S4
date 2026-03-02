# OpenFHE Workstation Access & Quick-Start

This guide helps you:
- verify OpenFHE is installed on the shared workstation
- set up your user environment
- build and run a minimal first project

Use this as a practical checklist for first-time setup.

## 1) Admin Check: Verify Installation

Before students start, confirm OpenFHE is installed in system paths.

| Component | Check command | Expected result |
|---|---|---|
| Headers | `ls /usr/local/include/openfhe` | Folders such as `core`, `pke`, `binfhe` |
| Libraries | `ls /usr/local/lib \| grep -i openfhe` | Files such as `libOPENFHEcore.so` |
| CMake config | `ls /usr/local/lib/OpenFHE` | `OpenFHEConfig.cmake` exists |

## 2) Student Setup: Environment Variables

To run binaries linked against shared OpenFHE libraries, add `/usr/local/lib` to your runtime library path.

Open your shell profile:

```bash
nano ~/.bashrc   # use ~/.zshrc if you use zsh
```

Add this line at the bottom:

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

Reload your profile:

```bash
source ~/.bashrc
```

> Important: if you skip `source ~/.bashrc`, your current terminal session will not see the updated `LD_LIBRARY_PATH`, and builds/runs may fail.

## 3) Create a Minimal OpenFHE Project

Do not build inside the OpenFHE source tree. Use a clean standalone folder.

### Step A: Initialize project files

```bash
mkdir -p ~/my_fhe_project
cd ~/my_fhe_project
touch main.cpp CMakeLists.txt
```

### Step B: Add `main.cpp`

This simple BGV example verifies that OpenFHE is discoverable and linkable.

```cpp
#include "pke/openfhe.h"
#include <iostream>

using namespace lbcrypto;

int main() {
    // 1) Configure a small BGV context
    CCParams<CryptoContextBGVRNS> parameters;
    parameters.SetMultiplicativeDepth(2);
    parameters.SetPlaintextModulus(65537);

    // 2) Construct the crypto context
    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);

    // 3) Enable basic capabilities
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);

    std::cout << "OpenFHE is working: CryptoContext generated successfully." << std::endl;
    return 0;
}
```

### Step C: Add `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.16)
project(OpenFHE_Student_Project)

# OpenFHE requires C++17+
set(CMAKE_CXX_STANDARD 17)

# 1) Locate OpenFHE install
find_package(OpenFHE REQUIRED)

# 2) Define executable
add_executable(fhe_test main.cpp)

# 3) Include OpenFHE headers for this workstation layout
target_include_directories(fhe_test PRIVATE
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe
)

# 4) Compile/link flags
target_compile_options(fhe_test PRIVATE -fopenmp)
target_link_libraries(fhe_test OPENFHEpke OPENFHEcore OPENFHEbinfhe -fopenmp)
```

## 4) Build and Run

Use a separate build directory:

```bash
mkdir -p build
cd build
cmake .. -DOpenFHE_DIR=/usr/local/lib/OpenFHE
make -j"$(nproc)"
./fhe_test
```

## 5) Troubleshooting

- **`Package OpenFHE not found`**
  - Try:
    ```bash
    cmake .. -DOpenFHE_DIR=/usr/local/lib/OpenFHE
    ```
- **Runtime linker errors (`libOPENFHE...so` not found)**
  - Re-check `LD_LIBRARY_PATH` and run `source ~/.bashrc` again.
  - This is the most common setup miss after editing `~/.bashrc`.
- **Permission concerns**
  - You do not need `sudo` to compile your own projects. `sudo` is only needed for system install steps. (under most cases you won't have sudo access to the workstations )

## 6) Useful Paths and References

- Shared examples on workstation:
  - `/usr/local/share/openfhe/examples` (may not exist on all workstations)
  - https://github.com/openfheorg/openfhe-development/tree/main/src/pke/examples (if shared examples are not there)
- Official OpenFHE documentation:
  - https://openfhe-development.readthedocs.io/en/latest/

## 7) Mini One-File C++ Smoke Test

If you want a quick sanity check, use this tiny single-file example.

Create `mini_openfhe.cpp`:

```cpp
#include "pke/openfhe.h"
#include <iostream>

using namespace lbcrypto;

int main() {
    CCParams<CryptoContextBGVRNS> params;
    params.SetMultiplicativeDepth(1);
    params.SetPlaintextModulus(65537);

    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);

    std::cout << "OpenFHE mini test OK" << std::endl;
    return 0;
}
```

Compile and run with CMake:

```bash
mkdir -p mini_build && cd mini_build
cat > CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(OpenFHEMiniTest)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenFHE REQUIRED)
add_executable(mini_openfhe ../mini_openfhe.cpp)
target_include_directories(mini_openfhe PRIVATE
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe
)
target_compile_options(mini_openfhe PRIVATE -fopenmp)
target_link_libraries(mini_openfhe OPENFHEpke OPENFHEcore OPENFHEbinfhe -fopenmp)
EOF
cmake .. -DOpenFHE_DIR=/usr/local/lib/OpenFHE
make -j"$(nproc)"
./mini_openfhe
```