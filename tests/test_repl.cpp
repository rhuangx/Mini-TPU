/// Plain line mode: echoes each command, skips blank lines, and stops at quit.

#include <sstream>
#include <string>

#include "test_framework.h"
#include "ui/repl.h"

TEST(repl_plain_mode_echoes_and_stops_at_quit) {
    Tpu tpu;
    Shell shell(tpu);
    std::istringstream in("step\n\nload /nonexistent/prog.s\nquit\nstep\n");
    std::ostringstream out;

    run_plain(shell, in, out);
    const std::string expected =
        "tpu> step\n"
        "error: unknown command 'step': the terminal takes load FILE and quit\n"
        "tpu> load /nonexistent/prog.s\n"
        "/nonexistent/prog.s: error: cannot open file\n"
        "tpu> quit\n";
    CHECK_EQ(out.str(), expected);
}
