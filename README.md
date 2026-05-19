# Introduction
Creates a section of the hat monotile tiling as described in *An aperiodic monotile*, https://doi.org/10.5070/C64163843
and based on the code at https://github.com/isohedral/hatviz/, and outputs the unique vertices of the section to a text
file.

# Usage
Run
```bash
./makemonotile [-l level: int default 0] [-w] [-o fname: str default monopts.txt]
```

# Build Instructions
For a release build (recommended)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

To enable visualization with raylib (causes a download of raylib)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMONOTILE_VISUAL=ON
```
