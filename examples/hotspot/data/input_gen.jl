#!/usr/bin/env -S julia --project

function create_experiment(n_rows, n_columns, temp_file, power_file)
    begin
        temp = fill(30.0f0, n_rows * n_columns)
        write(temp_file, temp)
    end

    begin
        power = zeros(Float32, n_rows, n_columns)
        power[(n_rows÷4):(3n_rows÷4), (n_columns÷4):(3n_columns÷4)] .= 0.5
        power = reshape(power', n_rows * n_columns)
        write(power_file, power)
    end
end

if size(ARGS) != (4,)
    println(stderr, "Usage: $PROGRAM_FILE <grid height> <grid width> <temp file> <power file>")
    exit(1)
end

create_experiment(parse(Int, ARGS[1]), parse(Int, ARGS[2]), ARGS[3], ARGS[4])