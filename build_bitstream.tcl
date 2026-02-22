# Vivado Tcl script — Build bitstream for Siamese LSTM on Ultra96-V2
# Using short path to avoid Windows 260-char path limit
# Uses a SINGLE SmartConnect with 3 slave ports (avoids black-box OOC failures)
#
# Usage: In Vivado Tcl Console:
#   source C:/Users/lsy/Downloads/Projects/Capstone/prototype/build_bitstream.tcl

# Use short base path to avoid Windows path length issues
set base_dir     "C:/hlsprj"
set project_name "slstm"
set src_dir      "C:/Users/lsy/Downloads/Projects/Capstone/prototype"
set orig_ip_path "$src_dir/siamese_lstm_hls2/solution1/impl/ip"
set ip_repo_path "$base_dir/ip"
set part         "xczu3eg-sbva484-2-i"

# Create short directory
file mkdir $base_dir

# Step 0: Copy HLS IP to short path to avoid Windows 260-char filename limit
puts "Copying HLS IP to short path: $ip_repo_path"
file mkdir $ip_repo_path
set orig_ip_win [string map {/ \\} $orig_ip_path]
set ip_repo_win [string map {/ \\} $ip_repo_path]
catch {exec robocopy $orig_ip_win $ip_repo_win /E /NFL /NDL /NJH /NJS /NC /NS /NP}
puts "IP copy complete."

# Step 1: Create project in short path
create_project $project_name $base_dir/$project_name -part $part -force

# Step 2: Add HLS IP repository
set_property ip_repo_paths $ip_repo_path [current_project]
update_ip_catalog

# Step 3: Create block design
create_bd_design "design_1"

# Step 4: Add Zynq UltraScale+ MPSoC
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 zynq_ultra_ps_e_0

# Step 5: Run block automation on Zynq PS first (sets up clocks/resets)
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "0"} [get_bd_cells zynq_ultra_ps_e_0]

# Step 6: Configure Zynq PS - enable M_AXI_HPM0_FPD and S_AXI_HP0_FPD
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {0} \
    CONFIG.PSU__USE__S_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP3 {0} \
    CONFIG.PSU__USE__S_AXI_GP4 {0} \
    CONFIG.PSU__USE__S_AXI_GP5 {0} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {32} \
] [get_bd_cells zynq_ultra_ps_e_0]

# Step 7: Add HLS IP
create_bd_cell -type ip -vlnv xilinx.com:hls:siamese_lstm_top:1.0 siamese_lstm_top_0

# Step 8: Connect s_axilite control port (PS master -> HLS slave)
# This also connects HLS IP's ap_clk and ap_rst_n
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/zynq_ultra_ps_e_0/pl_clk0} \
    Clk_slave {/zynq_ultra_ps_e_0/pl_clk0} \
    Clk_xbar {/zynq_ultra_ps_e_0/pl_clk0} \
    Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} \
    Slave {/siamese_lstm_top_0/s_axi_control} \
    intc_ip {New AXI Interconnect} master_apm {0}} [get_bd_intf_pins siamese_lstm_top_0/s_axi_control]

# Step 9: Create ONE SmartConnect with 3 slave ports for all m_axi data ports
# (Previous approach created 3 separate SmartConnects which caused black-box errors)
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_data
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] [get_bd_cells axi_smc_data]

# Step 10: Connect HLS m_axi ports to SmartConnect slave ports
connect_bd_intf_net [get_bd_intf_pins siamese_lstm_top_0/m_axi_gmem0] [get_bd_intf_pins axi_smc_data/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins siamese_lstm_top_0/m_axi_gmem1] [get_bd_intf_pins axi_smc_data/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins siamese_lstm_top_0/m_axi_gmem2] [get_bd_intf_pins axi_smc_data/S02_AXI]

# Step 11: Connect SmartConnect master to Zynq HP0
connect_bd_intf_net [get_bd_intf_pins axi_smc_data/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]

# Step 12: Connect clocks and resets for SmartConnect and HP0
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_smc_data/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk]
connect_bd_net [get_bd_pins rst_ps8_0_96M/peripheral_aresetn] [get_bd_pins axi_smc_data/aresetn]

# Step 13: Assign addresses
assign_bd_address

# Step 14: Validate and save
validate_bd_design
save_bd_design

# Step 15: Create HDL wrapper
make_wrapper -files [get_files $base_dir/$project_name/$project_name.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse $base_dir/$project_name/$project_name.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v

# Step 16: Generate bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Step 17: Report result
puts ""
puts "Implementation status: [get_property STATUS [get_runs impl_1]]"

# Step 18: Copy files for PYNQ
file mkdir $src_dir/pynq_overlay
if {[file exists $base_dir/$project_name/$project_name.runs/impl_1/design_1_wrapper.bit]} {
    file copy -force $base_dir/$project_name/$project_name.runs/impl_1/design_1_wrapper.bit $src_dir/pynq_overlay/siamese_lstm.bit
    file copy -force $base_dir/$project_name/$project_name.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh $src_dir/pynq_overlay/siamese_lstm.hwh
    puts ""
    puts "============================================================"
    puts "SUCCESS! Output files for PYNQ:"
    puts "  $src_dir/pynq_overlay/siamese_lstm.bit"
    puts "  $src_dir/pynq_overlay/siamese_lstm.hwh"
    puts "============================================================"
} else {
    puts "ERROR: Bitstream not found. Check implementation logs."
}
