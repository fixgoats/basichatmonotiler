# Introduction
Creates a section of the hat monotile tiling as described in *An aperiodic monotile*, https://doi.org/10.5070/C64163843
and based on the code at https://github.com/isohedral/hatviz/, and outputs the unique vertices of the section to a text
file.

# Usage
Run
```bash
./makemonotile [-l level: number of times to iterate metatile construction algorithm. int default 0]
               [-p: draw unique vertices of the tiles. bool default false]
               [-t: draw tiles with edges. bool default false]
               [-o fname: name of the file containing the generated vertices. str default monopts.txt]
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
