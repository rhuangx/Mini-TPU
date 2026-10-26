/// Shell: the terminal's load and quit, and errors as text.

#include <filesystem>
#include <fstream>
#include <string>

#include "test_framework.h"
#include "ui/shell.h"

namespace {

// Two rows in, two rows out: the read takes 24 cycles, the write 24 more, and Halt issues on cycle 48.
const char* kCopy =
    ".host 0\n"
    "1 2 3 4\n"
    ".host 1\n"
    "-1 -2 -3 -4\n"
    "Read_Host_Memory host=0 ub=0 rows=2\n"
    "Write_Host_Memory ub=0 host=4 rows=2\n"
    "Halt\n";

std::string write_temp_program(const std::string& name, const std::string& source) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream file(path);
    file << source;
    return path.string();
}

}  // namespace

TEST(shell_loads_a_program) {
    Tpu tpu;
    Shell shell(tpu);
    const std::string path = write_temp_program("mini_tpu_shell_copy.s", kCopy);

    CHECK_EQ(shell.execute("load " + path), "loaded " + path + ": 3 instructions, 8 host bytes, 0 weight bytes");
    CHECK_EQ(tpu.program().code.size(), std::size_t{3});
    CHECK_EQ(shell.execute("load /nonexistent/prog.s"), std::string("/nonexistent/prog.s: error: cannot open file"));
    CHECK_EQ(shell.execute("load"), std::string("error: usage: load FILE"));
}

TEST(shell_terminal_takes_only_load_and_quit) {
    Tpu tpu;
    Shell shell(tpu);
    CHECK_EQ(shell.execute("step"),
             std::string("error: unknown command 'step': the terminal takes load FILE and quit"));
    CHECK_EQ(shell.execute("   "), std::string(""));

    CHECK(!shell.quit_requested());
    CHECK_EQ(shell.execute("quit"), std::string(""));
    CHECK(shell.quit_requested());
}
