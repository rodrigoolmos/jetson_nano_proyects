# Example of nitefury(artix XC7A200T-2FBG484E) and jetson orin nano
In this example, you can send traffic through PCIe from AXI Lite or AXI Full and blink some LEDs.

## [00_vec_add](Build Vivado project)
```plaintext
source <path>/top.tcl
```

## [01_reg_vs_no_reg](Comple c source)
```plaintext
mkdir build
cd build
cmake ..
make
./axi_lite
./axi_full
```