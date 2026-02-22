# Vitis HLS Tcl script — C simulation for Siamese LSTM
#
# Target: Ultra96-V2 (Zynq UltraScale+ ZU3EG, xczu3eg-sbva484-1-e)
#
# Usage (from fpga/ directory):
#   vitis_hls -f run_csim.tcl
#
# Or step by step in Vitis HLS GUI:
#   1. Create project with these settings
#   2. Run C Simulation
#   3. Run C Synthesis
#   4. Export IP (for Vivado block design)

# Project setup
open_project siamese_lstm_hls
set_top siamese_lstm_top
add_files hls/siamese_lstm.cpp
add_files -tb hls/siamese_lstm_tb.cpp

# Target device: Ultra96-V2
open_solution "solution1" -flow_target vivado
set_part {xczu3eg-sbva484-1-e}
create_clock -period 10 -name default

# Run C simulation
csim_design

# Uncomment to also run synthesis:
# csynth_design

# Uncomment to export IP for Vivado:
# export_design -format ip_catalog

exit
