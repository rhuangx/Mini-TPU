/// tpu_sim: the terminal loads programs. Start it with ./tpu.

#include <unistd.h>

#include <iostream>
#include <string>

#include "core/tpu.h"
#include "ui/repl.h"
#include "ui/shell.h"

int main(int argc, char** argv) {
    // An argument is the program to load.
    std::string program;
    if (argc > 1) {
        program = argv[1];
    }

    Tpu tpu;
    Shell shell(tpu);

    std::cout << "Mini-TPU (TPUv1). Type load FILE to load a program. quit exits.\n";

    if (!program.empty()) {
        const std::string load_command = "load " + program;
        std::cout << "tpu> " << load_command << "\n" << shell.execute(load_command) << "\n";
    }

    const bool interactive = isatty(STDIN_FILENO) != 0 && isatty(STDOUT_FILENO) != 0;
    if (interactive) {
        run_interactive(shell);
    } else {
        run_plain(shell, std::cin, std::cout);
    }
    return 0;
}
