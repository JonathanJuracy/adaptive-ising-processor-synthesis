module top (
    input  wire clk,

    output wire clk_slow,
    output wire update_en,
    output wire rand_bit,

    output wire spin_0,
    output wire spin_1,
    output wire spin_2,
    output wire spin_3,

    output wire best_update
);

    // ============================================================
    // FPGA proof of concept:
    // Digital emulation of four p-bits using pseudo-random LFSR noise.
    // Simplified dynamics inspired by Ising / Max-Cut optimization.
    //
    // Tang Nano 1K - 27 MHz clock
    //
    // CH0 -> clk_slow
    // CH1 -> update_en
    // CH2 -> rand_bit
    // CH3 -> spin_0
    // CH4 -> spin_1
    // CH5 -> spin_2
    // CH6 -> spin_3
    // CH7 -> best_update
    // ============================================================

    reg [31:0] div = 32'd0;
    reg [7:0]  lfsr = 8'b1010_1101;
    reg [3:0]  spin = 4'b0011;
    reg [7:0]  best_hold = 8'd0;

    wire update_tick;

    // Slower update rate to make the signal easier to observe
    // on the logic analyzer.
    assign update_tick = (div[17:0] == 18'h00000);

    // Visible update pulse.
    assign update_en = (div[17:10] == 8'h00);

    // Slow clock used as a visual reference signal.
    assign clk_slow = div[15];

    // Pseudo-random output bit.
    assign rand_bit = lfsr[0];

    // Digital outputs of the four p-bits.
    assign spin_0 = spin[0];
    assign spin_1 = spin[1];
    assign spin_2 = spin[2];
    assign spin_3 = spin[3];

    // Pulse indicating that a favorable configuration was reached.
    assign best_update = (best_hold != 8'd0);

    // Simple count of "cut edges" for a Max-Cut problem on a ring:
    // s0-s1, s1-s2, s2-s3, and s3-s0.
    wire edge01 = spin[0] ^ spin[1];
    wire edge12 = spin[1] ^ spin[2];
    wire edge23 = spin[2] ^ spin[3];
    wire edge30 = spin[3] ^ spin[0];

    wire [2:0] cut_score = edge01 + edge12 + edge23 + edge30;

    always @(posedge clk) begin
        div <= div + 32'd1;

        if (update_tick) begin
            // Update the LFSR state.
            if (lfsr == 8'd0) begin
                lfsr <= 8'b1010_1101;
            end else begin
                lfsr <= {
                    lfsr[6:0],
                    lfsr[7] ^ lfsr[5] ^ lfsr[4] ^ lfsr[3]
                };
            end

            // Visual probabilistic update of the p-bits.
            // The update combines neighbor-dependent behavior with
            // pseudo-random noise.
            spin[0] <= (spin[1] ^ lfsr[0]) ^ lfsr[4];
            spin[1] <= (spin[2] ^ lfsr[1]) ^ lfsr[5];
            spin[2] <= (spin[3] ^ lfsr[2]) ^ lfsr[6];
            spin[3] <= (spin[0] ^ lfsr[3]) ^ lfsr[7];

            // Generate a pulse when the configuration has three or
            // four cut edges. This represents a favorable low-energy
            // configuration for the Max-Cut formulation.
            if (cut_score >= 3'd3) begin
                best_hold <= 8'd255;
            end
        end

        // Stretch the best_update pulse so that it can be clearly
        // observed on the logic analyzer.
        if (best_hold != 8'd0) begin
            best_hold <= best_hold - 8'd1;
        end
    end

endmodule